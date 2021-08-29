import streamlit as st

# from streamlit.hashing import _CodeHasher
import pandas as pd
from ludwig.api import LudwigModel
import os
import matplotlib as plt
import seaborn as sns
import io
import zipfile
import json
import tempfile
import warnings
import os

ModelFiles = os.path.isfile("training_set_metadata.json")

import pandas as pd
import io
import base64

# import streamlit as st
import extra_streamlit_components as stx

###############

# For Download button (Johannes)
from functionforDownloadButtons import download_button
import os
import json

###############

st.set_option("deprecation.showPyplotGlobalUse", False)

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
)


os.chdir("/app/fraud-detection-via-ludwig/")
cwd = os.getcwd()
cwd

[x[0] for x in os.walk(cwd)]

# st.stop()

# stepper_bar

st.title("💳 Credit Card Fraud Detector")
# df = pd.read_csv("creditcard.csv")

# st.caption("Dataset: https://bit.ly/3D92jJ3 | Article: https://bit.ly/3mulZkn")
#
st.caption(
    "PRD: https://bit.ly/3gutmVr | Dataset: https://bit.ly/3D92jJ3 | Article: https://bit.ly/3mulZkn"
)

val = stx.stepper_bar(steps=["Train", "Predict"])

# val = stx.stepper_bar(steps=["Train", "Predict", "Download"])

# st.info(f"Phase #{val}")
#
# chosen_id = stx.tab_bar(
#     data=[
#         stx.TabBarItemData(id=1, title="ToDo", description=""),
#         stx.TabBarItemData(id=2, title="Done", description=""),
#         # stx.TabBarItemData(id=3, title="Overdue", description="Tasks missed out"),
#     ],
#     default=1,
# )

# val = stx.stepper_bar(steps=["Ready", "Get Set", "Go"])

with st.beta_expander("📝 - Todo ", expanded=True):
    st.write(
        """
    - Deploy to sharing, make sure it works!
    - Deploy to sharing, make sure it works!
    - Deploy to sharing, make sure it works!
    - Balance dataset (currently no fraud results)
    - Align both widgets in same expander
    - Assign widget values to output features (currently 10 epochs!)
    - Add gif

    """
    )
#

with st.beta_expander(" Done ", expanded=False):
    st.write(
        """
#
    - Amend code for ccard dataset
    - Add form

    """
    )

with st.beta_expander(" Later ", expanded=False):
    st.write(
        """
    - Amend download button

    """
    )


c30, c32 = st.beta_columns([1, 0.6])

with c30:

    st.markdown("")
    uploaded_file = st.file_uploader("Choose a training csv", key=2)

    if uploaded_file is not None:
        # file_container = st.beta_expander('See your training csv')
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        # file_container.write(df.info())

        import pandas as pd

        # df.columns = ["doc_text", "isFraud"]
        header_list = [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ]

        # df = pd.read_csv("/content/DescriptiveFile1KRows.csv")
        df = df.iloc[1:]
        # df.head(3)

        # DfPivotCodes = df.groupby(["isFraud"]).agg({"isFraud": ["count"]})
        # DfPivotCodes.columns = [
        #     "_".join(multi_index) for multi_index in DfPivotCodes.columns.ravel()
        # ]
        # DfPivotCodes = DfPivotCodes.reset_index()

        st.markdown("")

        with st.beta_expander("📝 - See your training data", expanded=False):

            st.dataframe(df, height=550)
            # DfPivotCodes

    else:
        st.success("☝️ Upload your CSV first")
        st.stop()


# with c32:

# with st.form(key="my_form"):
# text_input = st.text_input(label="Enter some text")

# def text_field(label, columns=None, **input_params):
#     c1, c2, c3 = st.beta_columns(columns or [1, 2, 4])
#     # Display field name with some alignment
#     c1.markdown("##")
#     c1.markdown(label)
#     # Sets a default key parameter to avoid duplicate key errors
#     input_params.setdefault("key", label)
#     # Forward text input parameters
#     return c2.text_input("", **input_params)

# with c1:

with st.form(key="my_form"):

    selectbox = st.selectbox(
        "Choose encoder",
        [
            "parallel_cnn",
            "stacked_cnn",
            "stacked_parallel_cnn",
            "rnn",
            "cnnrnn",
            "transformer",
        ],
    )
    # with c3:
    slider = st.slider("How many Epochs?", 2, 10)
    # with c5:
    # selectbox2 = st.selectbox("Level of representation", ["word", "char"])

    # Initialise model

    # Train model

    input_features = [
        {"name": "step", "type": "category"},
        {"name": "type", "type": "category"},
        {"name": "amount", "type": "category"},
        {"name": "nameOrig", "type": "category"},
        {"name": "oldbalanceOrg", "type": "category"},
        {"name": "newbalanceOrig", "type": "category"},
        {"name": "nameDest", "type": "category"},
        {"name": "oldbalanceDest", "type": "category"},
        {"name": "newbalanceDest", "type": "category"},
        {"name": "isFraud", "type": "category"},
        {"name": "isFlaggedFraud", "type": "category"},
    ]

    # Output_features
    output_features = [{"name": "isFraud", "type": "category"}]

    # input_features = [
    #     {
    #         "name": "doc_text",
    #         "type": "text",
    #         # "level": selectbox2,
    #         "encoder": selectbox,
    #     }
    # ]
    #
    # output_features = [{"name": "isFraud", "type": "category"}]

    # Config
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 1, "fc_size": 48},
        "training": {"epochs": 10},
    }

    model = LudwigModel(config)

    # config = {
    #     "input_features": input_features,
    #     "output_features": output_features,
    #     "combiner": {"type": "concat", "fc_size": 14},
    #     "training": {"epochs": slider},
    # }

    # model = LudwigModel(config)

    start_execution = st.form_submit_button(label="Submit")

    if start_execution:
        c1, c2, c3 = st.beta_columns([5, 5, 5])
        with c2:
            gif_runner = st.image("credit-card.gif")
            # result = run_model(args)
            gif_runner.empty()
            # display_output(result)


train_stats, _, _ = model.train(dataset=df)
# train_stats, _, _ = model.train(dataset=df)
eval_stats, _, _ = model.evaluate(dataset=df)

model.save("saved_model")

from zipfile import ZipFile
import urllib.request
import zipfile
from io import BytesIO

os.chdir("saved_model")

zipObj = ZipFile("sample.zip", "w")
zipObj.write("checkpoint")
zipObj.write("model_hyperparameters.json")
zipObj.write("model_weights.data-00000-of-00001")
zipObj.write("model_weights.index")
zipObj.write("training_set_metadata.json")

# close the Zip File
zipObj.close()

ZipfileDotZip = "sample.zip"

# CSVButton2 = download_button(ZipfileDotZip, "Model2.zip", "Download CSV")
# st.table(inner_joinONEnotTWOStyled)

with open(ZipfileDotZip, "rb") as f:
    bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
        Download last model weights\
    </a>"
st.markdown(href, unsafe_allow_html=True)

# os.chdir("C:/Users/Charly/Desktop/LudwigNew")
os.chdir("/app/fraud-detection-via-ludwig/")

cwd = os.getcwd()
cwd

c1, c2, c3 = st.beta_columns(3)

with c1:
    Accuracy = eval_stats["isFraud"]["accuracy"]
    type(Accuracy)
    AccuracyPercentage = "{:.2%}".format(Accuracy)
    st.subheader("Accuracy:")
    st.subheader(AccuracyPercentage)

with c2:
    loss = eval_stats["isFraud"]["loss"]
    type(loss)
    lossPercentage = "{:.2%}".format(loss)
    st.subheader("Loss:")
    st.subheader(lossPercentage)

with c3:
    hits_at_k = eval_stats["isFraud"]["hits_at_k"]
    type(hits_at_k)
    hits_at_kPercentage = "{:.2%}".format(hits_at_k)
    st.subheader("hits_at_k:")
    st.subheader(hits_at_kPercentage)

from ludwig import visualize
import pandas as pd
import glob
import os

list_of_files = glob.glob(
    "results/api_experiment_run*/*"
)  # * means all if need specific format then *.csv

# results\api_experiment_run

latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
# st.write(latest_file)

import json

# with open('training_statistics.json') as f:
# with open('C:/Users/Charly/Desktop/LudwigNew/results/api_experiment_run_55/training_statistics.json') as f:
with open(latest_file) as f:
    data = json.load(f)
# st.write(data)

fig = visualize.learning_curves(
    [data],
    # output_directory="C:/Users/Charly/Desktop/LudwigNew/Images/LudwigImages",
    output_directory="/app/fraud-detection-via-ludwig/Images/LudwigImages",
    file_format="png",
)

# st.pyplot(fig, clear_figure=False)
# c1, c2 = st.beta_columns(2)
# with c1:
# st.write('TH_tagged_deduped_dec_2020')

# with st.beta_expander("📝 - Roadmap TO-DO'S TRAINING TAB", expanded=True):

st.image(
    "Images/LudwigImages/learning_curves_isfraud_accuracy.png",
    use_column_width=True,
)
# with c1:
st.image(
    "Images/LudwigImages/learning_curves_isfraud_loss.png",
    use_column_width=True,
)

# c1, c2 = st.beta_columns(2)
# with c1:
st.image(
    "Images/LudwigImages/learning_curves_isfraud_hits_at_k.png",
    use_column_width=True,
)
# with c1:
st.image(
    "Images/LudwigImages/learning_curves_combined_loss.png",
    use_column_width=True,
)

# st.pyplot()
# Needs more parameters
# st.pyplot(visualize.compare_classifiers_performance_subset([data]))

# c2.markdown(status.messages())
