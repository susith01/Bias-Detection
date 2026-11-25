"""Page 2: Dataset Selection, CSV Upload, Quality Profiling, and Sample Downloads."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from config import UPLOADS_DIR
from dataset.generate_datasets import DATASET_CATALOG, ensure_all_datasets
from modules.preprocessing import (
    apply_age_grouping,
    detect_sensitive_columns,
    inspect_dataset,
)
from modules.ui_helpers import (
    get_db,
    load_builtin_dataset_into_state,
    render_sidebar_status,
)

st.set_page_config(page_title="Upload Dataset | FairLens AI", layout="wide")
render_sidebar_status()

st.title("2. Dataset Selection & CSV Upload")
st.caption(
    "Choose one of the three benchmark datasets (clearly labeled synthetic vs historical) "
    "or upload your own CSV decision dataset. Unrelated datasets are always kept separate."
)

tab_builtin, tab_upload, tab_samples = st.tabs(
    ["Built-In Datasets", "Upload Custom CSV", "Sample CSV Downloads"]
)

with tab_builtin:
    for ds_name, meta in DATASET_CATALOG.items():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(f"{ds_name} [{meta['source_type'].upper()}]")
                st.write(meta["description"])
                st.caption(f"Domain: {meta['domain']} | Target Column: {meta['target_column']}")
            with c2:
                if st.button(f"Load {ds_name}", key=f"load_{ds_name}", use_container_width=True):
                    df_loaded = load_builtin_dataset_into_state(ds_name)
                    db = get_db()
                    profile = inspect_dataset(df_loaded)
                    res = db.save_dataset_metadata(
                        {
                            "dataset_name": ds_name,
                            "source_type": meta["source_type"],
                            "domain": meta["domain"],
                            "row_count": profile["row_count"],
                            "column_count": profile["column_count"],
                            "target_column": meta["target_column"],
                            "sensitive_columns": profile["candidate_sensitive"],
                            "missing_summary": profile["missing_summary"],
                            "duplicate_count": profile["duplicate_count"],
                        }
                    )
                    st.session_state["active_dataset_id"] = res["data"].get("id")
                    st.success(f"Loaded {ds_name} ({len(df_loaded)} rows).")

with tab_upload:
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        profile = inspect_dataset(df_up)
        default_target = (
            profile["candidate_targets"][0]
            if profile["candidate_targets"]
            else df_up.columns[-1]
        )
        with st.form("confirm_upload_form"):
            ds_label = st.text_input("Dataset Name", value=uploaded.name.replace(".csv", ""))
            target_col = st.selectbox("Binary Target Column", df_up.columns.tolist(), index=df_up.columns.tolist().index(default_target))
            domain_label = st.text_input("Decision Domain", value="Custom AI Decision System")
            confirm = st.form_submit_button("Activate Uploaded Dataset")
            if confirm:
                sens = detect_sensitive_columns(df_up)
                if sens.get("age"):
                    df_up = apply_age_grouping(df_up, age_col=sens["age"])
                save_path = UPLOADS_DIR / uploaded.name
                df_up.to_csv(save_path, index=False)

                st.session_state["active_dataset_name"] = ds_label
                st.session_state["active_dataset_source"] = "uploaded"
                st.session_state["active_df"] = df_up
                st.session_state["target_column"] = target_col
                st.session_state["sensitive_map"] = sens
                st.session_state["trained_bundles"] = {}

                db = get_db()
                db.upload_file("fairlens-datasets", save_path, uploaded.name)
                res = db.save_dataset_metadata(
                    {
                        "dataset_name": ds_label,
                        "source_type": "uploaded",
                        "domain": domain_label,
                        "row_count": profile["row_count"],
                        "column_count": profile["column_count"],
                        "target_column": target_col,
                        "sensitive_columns": profile["candidate_sensitive"],
                        "missing_summary": profile["missing_summary"],
                        "duplicate_count": profile["duplicate_count"],
                        "storage_path": f"fairlens-datasets/{uploaded.name}",
                    }
                )
                st.session_state["active_dataset_id"] = res["data"].get("id")
                st.success(f"Activated uploaded dataset '{ds_label}' ({len(df_up)} rows).")

with tab_samples:
    paths = ensure_all_datasets(try_remote_uci=False)
    for ds_name, p in paths.items():
        data_bytes = Path(p).read_bytes()
        st.download_button(
            label=f"Download {ds_name} CSV",
            data=data_bytes,
            file_name=Path(p).name,
            mime="text/csv",
        )

active_df = st.session_state.get("active_df")
if active_df is not None:
    st.divider()
    st.subheader(f"Active Dataset Profile: {st.session_state.get('active_dataset_name')}")
    prof = inspect_dataset(active_df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", prof["row_count"])
    m2.metric("Columns", prof["column_count"])
    m3.metric("Total Missing Cells", prof["total_missing"])
    m4.metric("Duplicate Rows", prof["duplicate_count"])

    c_num, c_cat = st.columns(2)
    with c_num:
        st.write("**Numerical Columns:**", ", ".join(prof["numeric_columns"]) or "None")
    with c_cat:
        st.write("**Categorical Columns:**", ", ".join(prof["categorical_columns"]) or "None")

    st.write("**Missing Values by Column:**")
    missing_df = pd.DataFrame(
        [{"Column": k, "Missing Count": v} for k, v in prof["missing_summary"].items()]
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.write("**Dataset Preview (First 25 Rows):**")
    st.dataframe(active_df.head(25), use_container_width=True)
