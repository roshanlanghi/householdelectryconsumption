import pptx

prs = pptx.Presentation('ADS Project PPT.pptx')
slide10 = prs.slides[9]

lines = [
    '1. Machine Learning Methodology (Research Paper): M. Brundage et al., "CRISP-ML(Q): A Process Model for Machine Learning Development," arXiv preprint, 2020. 🔗 https://arxiv.org/abs/2003.05155',
    '2. Data Science Framework: IBM, "CRISP-DM Methodology," 2024. 🔗 https://www.ibm.com/topics/crisp-dm',
    '3. Deep Learning Framework: TensorFlow & Keras Developers, "TensorFlow Core Documentation," 2024. 🔗 https://www.tensorflow.org/api_docs',
    '4. Machine Learning Library: scikit-learn Developers, "scikit-learn Documentation," 2024. 🔗 https://scikit-learn.org/stable/documentation.html',
    '5. Project Dataset Source: UCI Machine Learning Repository, "Individual Household Electric Power Consumption Dataset," 2024. 🔗 https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption',
    '6. Web Application & Visualization: Streamlit & Plotly Developers, "Streamlit Documentation," 2024. 🔗 https://docs.streamlit.io/'
]

for shape in slide10.shapes:
    if shape.has_text_frame and 'DEPARTMENT' not in shape.text and 'References' not in shape.text and shape.text.strip():
        tf = shape.text_frame
        tf.clear()
        for p_idx, line in enumerate(lines):
            p = tf.paragraphs[0] if p_idx == 0 else tf.add_paragraph()
            p.text = line

prs.save('ADS Project PPT.pptx')
print('Successfully updated Slide 10 references!')
