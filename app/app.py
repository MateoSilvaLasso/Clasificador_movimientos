import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
from mediapipe.tasks import python as p
from mediapipe.tasks.python import vision
from sklearn.impute import SimpleImputer
import streamlit as st

modelo = joblib.load("models/modelo_clasificacion_actividades.pkl")


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def normalizar_coordenadas(df):
    cadera_derecha = df[df['landmark_index'] == 23][['x', 'y', 'z']].mean()
    cadera_izquierda = df[df['landmark_index'] == 24][['x', 'y', 'z']].mean()
    cadera_centro = (cadera_derecha + cadera_izquierda) / 2

    df['x'] -= cadera_centro['x']
    df['y'] -= cadera_centro['y']
    df['z'] -= cadera_centro['z']

    hombro_derecho = df[df['landmark_index'] == 11][['x', 'y', 'z']].mean()
    hombro_izquierdo = df[df['landmark_index'] == 12][['x', 'y', 'z']].mean()
    torso_tamano = np.linalg.norm(hombro_derecho - hombro_izquierdo)

    df['x'] /= torso_tamano
    df['y'] /= torso_tamano
    df['z'] /= torso_tamano

    return df

def filtrar_datos(df):
    # Convertir las columnas x, y, z a tipo float
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)

    # Aplicar el filtro gaussiano
    df['x'] = gaussian_filter1d(df['x'], sigma=2)
    df['y'] = gaussian_filter1d(df['y'], sigma=2)
    df['z'] = gaussian_filter1d(df['z'], sigma=2)

    return df

def calcular_angulo(p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angulo = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angulo)

def extract_landmarks_to_dataframe(detection_result):
    # Crear una lista para almacenar las coordenadas de cada landmark
    landmarks = []
    for pose_landmark in detection_result.pose_landmarks:
        for landmark in pose_landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

   
    df_landmarks = pd.DataFrame(landmarks, columns=['x', 'y', 'z', 'visibility'])

    
    df_landmarks['landmark_index'] = df_landmarks.index
    return df_landmarks

def generar_caracteristicas(df):
    articulaciones_clave = [11, 12, 13, 14, 15, 16, 23, 24]

    velocidades = []
    angulos_codo_derecho = []
    angulos_codo_izquierdo = []
    angulos_tronco = []

    for articulacion in articulaciones_clave:
        actual = df[df['landmark_index'] == articulacion]
        if len(actual) > 1:
            vel_x = actual['x'].iloc[-1] - actual['x'].iloc[-2]
            vel_y = actual['y'].iloc[-1] - actual['y'].iloc[-2]
            vel_z = actual['z'].iloc[-1] - actual['z'].iloc[-2]
            velocidad = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            velocidades.append(velocidad)

    hombro_derecho = df[df['landmark_index'] == 11][['x', 'y', 'z']].values
    codo_derecho = df[df['landmark_index'] == 13][['x', 'y', 'z']].values
    muneca_derecha = df[df['landmark_index'] == 15][['x', 'y', 'z']].values

    if hombro_derecho.size > 0 and codo_derecho.size > 0 and muneca_derecha.size > 0:
        angulos_codo_derecho.append(calcular_angulo(hombro_derecho[0], codo_derecho[0], muneca_derecha[0]))

    hombro_izquierdo = df[df['landmark_index'] == 12][['x', 'y', 'z']].values
    codo_izquierdo = df[df['landmark_index'] == 14][['x', 'y', 'z']].values
    muneca_izquierda = df[df['landmark_index'] == 16][['x', 'y', 'z']].values

    if hombro_izquierdo.size > 0 and codo_izquierdo.size > 0 and muneca_izquierda.size > 0:
        angulos_codo_izquierdo.append(calcular_angulo(hombro_izquierdo[0], codo_izquierdo[0], muneca_izquierda[0]))

    if hombro_derecho.size > 0 and hombro_izquierdo.size > 0:
        cadera_centro = (df[df['landmark_index'] == 23][['x', 'y', 'z']].values +
                         df[df['landmark_index'] == 24][['x', 'y', 'z']].values) / 2
        angulos_tronco.append(calcular_angulo(hombro_derecho[0], cadera_centro[0], hombro_izquierdo[0]))

    caracteristicas =  np.array([
        np.mean(velocidades), np.std(velocidades),
        np.mean(angulos_codo_derecho), np.std(angulos_codo_derecho),
        np.mean(angulos_codo_izquierdo), np.std(angulos_codo_izquierdo),
        np.mean(angulos_tronco), np.std(angulos_tronco)
    ])

    

    return caracteristicas



st.set_page_config(
    page_title="Sistema de Detección de Poses",
    layout="wide"
)


st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🤸 Sistema de Detección de Poses")


with st.sidebar:
    st.header("⚙️ Configuración")
    
    
    model_path = st.text_input(
        "Ruta del modelo pose_landmarker_heavy.task",
        value="posemodels/pose_landmarker_heavy.task"
    )
    
    


if 'running' not in st.session_state:
    st.session_state.running = False


col1, col2 = st.columns([3, 1])

with col1:
    
    video_placeholder = st.empty()

with col2:
    st.subheader("🎮 Controles")
    
    
    if not st.session_state.running:
        if st.button("▶️ Iniciar Sistema"):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("⏹️ Detener Sistema"):
            st.session_state.running = False
            st.rerun()
    
    
    st.markdown("### 🎯 Predicción Actual")
    prediction_placeholder = st.empty()


if st.session_state.running:
    try:
        
        base_options = p.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)

        
        cap = cv2.VideoCapture(0)
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al acceder a la cámara")
                break

            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            
            if detection_result.pose_landmarks:
                for pose_landmark in detection_result.pose_landmarks:
                    for point in pose_landmark:
                        x = int(point.x * frame.shape[1])
                        y = int(point.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                
                df_landmarks = extract_landmarks_to_dataframe(detection_result)
                df_landmarks = normalizar_coordenadas(df_landmarks)
                df_landmarks = filtrar_datos(df_landmarks)
                df_landmarks.fillna(0, inplace=True)

                caracteristicas = generar_caracteristicas(df_landmarks)
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                caracteristicas = imputer.fit_transform([caracteristicas])
                prediccion = modelo.predict(caracteristicas)[0]

               
                with prediction_placeholder.container():
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style='text-align: center; color: #1f77b4;'>{prediccion}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.running = False

else:
    
    with video_placeholder:
        st.info("Sistema detenido. Presiona 'Iniciar Sistema' para comenzar.")
