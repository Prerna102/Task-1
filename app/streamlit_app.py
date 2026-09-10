import os
import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PaddleOCR",
    page_icon="📄",
    layout="centered",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #4F46E5;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f7f7f7;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title"> PaddleOCR</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image or document and extract text using PaddleOCR'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Upload section
# --------------------------------------------------

st.subheader("Upload Image or File")


uploaded_file = st.file_uploader(
    "Choose an image or file",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tiff",
        "pdf",
    ],
    help="Upload an image or PDF document for OCR.",
)


# --------------------------------------------------
# Display uploaded file
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"Selected file: {uploaded_file.name}"
    )

    file_extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )


    # Display image preview
    if file_extension in [
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tiff",
    ]:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True,
        )

    else:

        st.info(
            "PDF/document selected. "
            "The file will be sent to the OCR service."
        )


# --------------------------------------------------
# OCR button
# --------------------------------------------------

if uploaded_file is not None:

    run_ocr = st.button(
        "🔍 Run OCR",
        type="primary",
        use_container_width=True,
    )


    if run_ocr:

        with st.spinner(
            "Running PaddleOCR..."
        ):

            try:

                # Reset file position
                uploaded_file.seek(0)


                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type,
                    )
                }


                # Send file to FastAPI
                response = requests.post(
                    f"{API_URL}/api/ocr",
                    files=files,
                    timeout=300,
                )


                # Check response
                if response.status_code != 200:

                    try:
                        error_data = response.json()

                        error_message = error_data.get(
                            "detail",
                            "OCR processing failed.",
                        )

                    except Exception:

                        error_message = (
                            response.text
                            or "OCR processing failed."
                        )


                    st.error(
                        error_message
                    )

                else:

                    data = response.json()


                    st.success(
                        "OCR completed successfully!"
                    )


                    # --------------------------------------------------
                    # Confidence
                    # --------------------------------------------------

                    confidence = data.get(
                        "confidence",
                        0,
                    )


                    st.metric(
                        "OCR Confidence",
                        f"{confidence * 100:.2f}%",
                    )


                    # --------------------------------------------------
                    # Extracted text
                    # --------------------------------------------------

                    st.subheader(
                        "Extracted Text"
                    )


                    extracted_text = data.get(
                        "text",
                        "",
                    )


                    st.text_area(
                        "OCR Result",
                        value=extracted_text,
                        height=300,
                    )


                    # --------------------------------------------------
                    # Download text
                    # --------------------------------------------------

                    st.download_button(
                        label="⬇️ Download Text",
                        data=extracted_text,
                        file_name=(
                            f"{uploaded_file.name}"
                            ".txt"
                        ),
                        mime="text/plain",
                        use_container_width=True,
                    )


                    # --------------------------------------------------
                    # Database ID
                    # --------------------------------------------------

                    st.caption(
                        f"OCR Result ID: {data.get('id')}"
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI OCR service."
                )

                st.info(
                    f"Make sure the API is running at: {API_URL}"
                )


            except requests.exceptions.Timeout:

                st.error(
                    "OCR processing timed out. "
                    "The document may be too large."
                )


            except Exception as exc:

                st.error(
                    f"Unexpected error: {str(exc)}"
                )


# --------------------------------------------------
# API health check
# --------------------------------------------------

st.divider()

with st.expander("Service Information"):

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )


        if response.status_code == 200:

            st.success(
                "FastAPI service is online."
            )

        else:

            st.warning(
                "FastAPI service returned an unexpected response."
            )

    except Exception:

        st.error(
            "FastAPI service is unavailable."
        )
