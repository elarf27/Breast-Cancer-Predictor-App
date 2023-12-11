import streamlit as st
import pickle5 as pickle
import pandas as pd
import numpy as np
from plotly import graph_objects
from streamlit_option_menu import option_menu

st.set_page_config(
        page_title= 'Web App Name',
        page_icon= ':dog:',
        layout='wide',
        initial_sidebar_state='expanded'
        
    )

selected = option_menu(
    menu_title=None,
    icons=['house', 'cpu', 'info-lg', 'envelope'],
    options=['Home', 'Predictor', 'About the predictor', 'Contact'],
    default_index=0,
    orientation='horizontal', 
    styles={
        'nav-link':{
            'font-size' : '18px',
            'margin': '0px',
            '--hover-color': '#0e1117',
        },
        'nav-link-selected':{'background-color': '#E75480'},
    },
    
)

def header(text):
     text_bold = f'<strong>{text}</strong>'
     st.markdown(f'<p style="background-color:#E75480;color:#FFFF;font-size:36px;border-radius:2%;">{text_bold}</p>', unsafe_allow_html=True)
def title(text):
    text_bold = f'<strong>{text}</strong>'
    st.markdown(f'<p style="background-color:#E75480;color:#FFFF;font-size:45px;border-radius:2%;">{text_bold}</p>', unsafe_allow_html=True)

if selected == 'Home':
    title('Breast Cancer Predictor')
    st.markdown('<span style="font-size:150%;">_Advancing awareness against_ <span style="color:#E75480;">_breast cancer_</span> _using_ <span style="color:#E75480;">_artificial intelligence_</span></span>.', unsafe_allow_html=True)
    st.markdown('Greetings and welcome to <span style="color:#E75480;">Breast Cancer Predictor</span>, a cutting-edge platform at the forefront of breast cancer prediction. With an unwavering commitment to precision and reliability, our app harnesses the power of artificial intelligence to deliver personalized insights into the probability of breast cancer being <span style="color:#E75480;">malicious</span> or <span style="color:#E75480;">benign</span>. At Breast Cancer Predictor, we recognize the significance of proactive health management and the importance of informed decision-making. Guided by the principles of transparency and innovation, our application stands as a beacon of technological prowess, utilizing state-of-the-art algorithms trained on a real breast cancer diagnostic data set.', unsafe_allow_html=True)
    
    st.subheader('Our Commitment')
    st.markdown('Accuracy: Experience the pinnacle of predictive analytics, providing you with reliable assessments of breast cancer probability.')
    st.markdown('Empathy: We understand the gravity of health-related concerns. Our mission is to empower you with knowledge, fostering a sense of control and assurance.')
    st.markdown('Privacy: Rest easy knowing that your data is handled with the utmost confidentiality and security. Your well-being is our priority.')

    st.subheader('Navigating Your Journey')
    st.markdown('Explore the user-friendly interface, seamlessly designed to guide you through the prediction process. Whether you seek peace of mind, early detection, or a tool for proactive health management,Breast Cancer Predictor is your trusted companion on this crucial path.')

    st.subheader('Acting Responsibley')
    st.markdown('At Breast Cancer Predictor, we stand at the intersection of technology and healthcare, committed to providing you with valuable insights into the probability of breast cancer. It is crucial to emphasize, however, that the predictions offered by our advanced model <span style="color:red;">are not</span> intended to replace the expertise and personalized care provided by healthcare professionals.</span>', unsafe_allow_html=True)
    st.markdown('While our application serves as a powerful tool for early awareness and informed decision-making, <span style="color:red;">it is imperative</span> to consult with <span style="color:red;">qualified healthcare professionals</span> for a comprehensive diagnosis. The nuances of individual health conditions require the discerning eye of experienced medical practitioners who can consider a multitude of factors beyond the scope of our predictive model.</span>', unsafe_allow_html=True)
    st.markdown('At Breast Cancer Predictor, we advocate for a collaborative approach to healthcare, where our predictions act as a supplementary resource, complementing the invaluable expertise of healthcare professionals. Your well-being is our utmost priority, and we encourage you to seek professional medical advice for a thorough assessment tailored to your unique health profile.')

elif selected == 'About the predictor':
    
    header('About the Predictor')
    
    st.markdown('Welcome to our Breast Cancer Predictor, a powerful tool designed to assist in the early diagnosis of breast tumors. Our predictor utilizes advanced machine learning techniques trained on a dataset that captures essential features computed from digitized images of fine needle aspirates, providing valuable insights into cell nuclei characteristics.')
    st.subheader('Key Features')
    key_features = [
        'Radius: Mean distance from the center to points on the perimeter.',
        'Texture: Standard deviation of gray-scale values.',
        'Perimeter: Boundary length of the cell nucleus.',
        'Area: Total area covered by the cell nucleus.',
        'Smoothness: Local variation in radius lengths.',
        'Compactness: Perimeter^2 / Area - 1.0.',
        'Concavity: Severity of concave portions of the contour.',
        'Concave Points: Number of concave portions of the contour.',
        'Symmetry: Symmetry of the cell nucleus.',
        'Fractal Dimension: "Coastline approximation" - 1.'
    ]

    for feature in key_features:
        st.write(feature)

    st.subheader("Dataset Origin")

    st.markdown('The dataset used for training comes from a groundbreaking study that employed interactive image processing techniques. A linear-programming-based inductive classifier was utilized for accurate breast tumor diagnosis. Images of fine needle aspirates were digitized, and an interactive interface allowed precise analysis of nuclear size, shape, and texture.')
    
    st.subheader('Achievements')
    
    st.markdown('The system achieved a 97% accuracy in distinguishing between benign and malignant samples using a single separating plane on three features: mean texture, worst area, and worst smoothness.')
    
    st.markdown('While our predictor is a valuable tool, it is essential to note that it is not a substitute for professional medical advice. We encourage users to consult with qualified healthcare professionals for comprehensive diagnoses and personalized care.')
    st.markdown('Thank you for choosing our Breast Cancer Predictor. Together, let us empower proactive health management and contribute to early detection.')

elif selected == 'Contact':

    header("Contact Us")

    st.markdown(
        f"""
        If you have any questions, feedback, or need further assistance, feel free to reach out to us. We are here to help and provide support. You can contact us through the following channels:

         :email: **Email:** a.elaarfaoui12@gmail.com

         :telephone_receiver: **Phone:** +31 685-568-983

        We value your input and are committed to ensuring your experience with our Breast Cancer Predictor is positive and informative.

        Thank you for choosing our service, and we look forward to assisting you.
        """,
        unsafe_allow_html=True
    )

elif selected == 'Predictor':
    
    def get_data():
        data = pd.read_csv('/Users/abdou/Desktop/VsCode/project/data/data.csv')
        data = data.drop(['Unnamed: 32', 'id'], axis = 1)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

        return data
    
    def sidebar():

        st.sidebar.header("Cell Nuclei Measurements")

        st.markdown("<style>[data-testid=stSidebar] {background-color: #E75480;}</style>", unsafe_allow_html=True)

        
    
        data = get_data()

        indicator = [
        'Radius (mean)',
        'Texture (mean)',
        'Perimeter (mean)',
        'Area (mean)',
        'Smoothness (mean)',
        'Compactness (mean)',
        'Concavity (mean)',
        'Concave points (mean)',
        'Symmetry (mean)',
        'Fractal dimension (mean)',
        'Radius (se)',
        'Texture (se)',
        'Perimeter (se)',
        'Area (se)',
        'Smoothness (se)',
        'Compactness (se)',
        'Concavity (se)',
        'Concave points (se)',
        'Symmetry (se)',
        'Fractal dimension (se)', 
        'Radius (worst)', 
        'Texture (worst)', 
        'Perimeter (worst)', 
        'Area (worst)',
        'Smoothness (worst)', 
        'Compactness (worst)', 
        'Concavity (worst)', 
        'Concave points (worst)',
        'Symmetry (worst)', 
        'Fractal dimension (worst)'
    ]

        hold = {}

        for i, j in zip(indicator, data.drop(['diagnosis'], axis = 1).columns):
            hold[j] = st.sidebar.number_input(label=i, min_value=float(0), max_value=None, value=float(data[j].mean()))

        return hold
    
    def scale(patient_dic):
    
        data = get_data()

        scaled_dic = {}

        for i, j  in patient_dic.items():
            maximum = data.drop(['diagnosis'], axis = 1)[i].max()
            minimum = data.drop(['diagnosis'], axis = 1)[i].min()
            scaled = (j - minimum) / (maximum - minimum)

            scaled_dic[i] = scaled

        return scaled_dic
    
    def radar_chart(patient_data):
    
        patient_data = scale(patient_data)
        
        categories = [
          'Radius',
          'Texture', 
          'Perimeter',
          'Area',
          'Smoothness',
          'Compactness',
          'Concavity',
          'Concave Points',
          'Symmetry',
          'Fractal Dimension'
     ]
        
        fig = graph_objects.Figure()

        fig.add_trace(graph_objects.Scatterpolar(
          r = [
               patient_data['radius_mean'],
                patient_data['texture_mean'],
                patient_data['perimeter_mean'],
                patient_data['area_mean'],
                patient_data['smoothness_mean'], 
                patient_data['compactness_mean'],
                patient_data['concavity_mean'], 
                patient_data['concave points_mean'], 
                patient_data['symmetry_mean'],
                patient_data['fractal_dimension_mean']
          ],
          theta=categories,
          fill='toself',
          name = 'Mean Value'
     ))

        fig.add_trace(graph_objects.Scatterpolar(
          r = [
               patient_data['radius_se'],
                patient_data['texture_se'],
                patient_data['perimeter_se'],
                patient_data['area_se'],
                patient_data['smoothness_se'], 
                patient_data['compactness_se'],
                patient_data['concavity_se'], 
                patient_data['concave points_se'], 
                patient_data['symmetry_se'],
                patient_data['fractal_dimension_se']
          ],
          theta=categories,
          fill='toself',
          name='Standard Error'
     ))

        fig.add_trace(graph_objects.Scatterpolar(
          r = [
               patient_data['radius_worst'],
                patient_data['texture_worst'],
                patient_data['perimeter_worst'],
                patient_data['area_worst'],
                patient_data['smoothness_worst'], 
                patient_data['compactness_worst'],
                patient_data['concavity_worst'], 
                patient_data['concave points_worst'], 
                patient_data['symmetry_worst'],
                patient_data['fractal_dimension_worst']
          ],
          theta=categories,
          fill='toself',
          name='Worst Value'
     ))

        fig.update_layout(
          polar = dict(
               radialaxis = dict(
                    visible = True,
                    range=[0,1],
                    gridcolor='black'
               )
          ),
          showlegend=True
     )
        fig.update_polars(radialaxis_color='black')

        return fig
    
    def get_predictions(patient_data):
        
        model = pickle.load(open("model/model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
        patient_array = np.array(list(patient_data.values())).reshape(1, -1)
  
        patient_array_scaled = scaler.transform(patient_array)
  
        prediction = model.predict(patient_array_scaled)
  
        st.subheader("Cell cluster prediction")
        st.write("The cell cluster is:")
  
        if prediction[0] == 0:
            st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
        else:
            st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
    
        prob_benign = np.round(model.predict_proba(patient_array_scaled)[0][0] * 100, 2)
        prob_malignant = np.round(model.predict_proba(patient_array_scaled)[0][1] * 100, 2)

        if prob_benign == 100.00 and prob_malignant == 0.00:
            st.write(f'Probability of being <span style="color:#01DB4B;">benign</span>: >99.0%', unsafe_allow_html=True)
            st.write(f'Probability of being <span style="color:#ff4b4b;">malignant</span>: <1.0%', unsafe_allow_html=True)
        elif prob_benign == 0.00 and prob_malignant == 100.00:
            st.write(f'Probability of being <span style="color:#01DB4B;">benign</span>: <1.0%', unsafe_allow_html=True)
            st.write(f'Probability of being <span style="color:#ff4b4b;">malignant</span>: >99.0%', unsafe_allow_html=True)
        else:
            st.write(f'Probability of being <span style="color:#01DB4B;">benign</span>: {prob_benign}%', unsafe_allow_html=True)
            st.write(f'Probability of being <span style="color:#ff4b4b;">malignant</span>: {prob_malignant}%', unsafe_allow_html=True)
    
    def main():

        with open('style/layout.css') as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
        patient_data = sidebar()

        with st.container():
            st.write('This tool predicts if a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also do so yourself by updating the measurements in the sidebar.')

        col_1, col_2 = st.columns([4, 1])

        with col_1:
            plot = radar_chart(patient_data)
            st.plotly_chart(plot, theme=None)
        with col_2:
            get_predictions(patient_data)

    if __name__ == '__main__':
        main()