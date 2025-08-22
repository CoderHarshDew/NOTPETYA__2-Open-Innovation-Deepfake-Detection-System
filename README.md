# NOTPETYA__2-Open-Innovation-Deepfake-Detection-System
A **Deepfake Detection System**, an open innovation project by **NOTPETYA__2**  

---

## 👨‍💻 Team – NOTPETYA__2

- Harsh Dewangan (Leader) – Backend, UI, Model Training, Documentation

- Kavya Dewangan – Backend, Model Training, UI, Documentation

- Manya Pansari – Documentation, Frontend

- Chandrakant Dariyana – Documentation, Frontend

---

## 🚨 Problem Overview  
Deepfake technology is rapidly growing and causes several issues:  
- **Social Problems**: Misinformation, harassment, defamation, loss of trust.  
- **Legal & Ethical Problems**: Privacy violations, blackmail, cyberbullying, weak legal frameworks.  
- **Technological Problems**: Constantly improving deepfakes make detection harder.  

---

## 🎯 Our Solution (Mitigation)
- **Victims**: Prove their credibility.  
- **Public**: Verify if information is true or false.  
- **Authorities**: Detect early and stop spread.  
- **Impact**: Prevent harm & protect reputation.  

## 🎯 Our Solution (Process)  
- **Upload** : user gives an images 
- **Preprocess** : system extracts frames for analysis
- **Feature check** : AI looks for unnatural patterns or inconsistencies
- **AI Analysis** : model compares with real vs fake patterns
- **Result** : shows if content is Real or Fake with confidence.

---

## 🛠️ Tech Stack  
- **Language**: Python  
- **Frameworks**: TensorFlow, Keras, Streamlit  
- **Libraries**: NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, PIL  
- **Dataset**: Self-made dataset (Real vs Fake images) — not publicly shared due to privacy reasons

---

## ⚙️ Installation & Usage 

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/deepfake-detection.git
   cd deepfake-detection


2. Install dependencies:

        pip install -r requirements.txt

3. Run training (optional, if you want to retrain the model):

        python deepfake_detection_system_web_app.py

4. Run the Streamlit app:

        streamlit run app.py

## 📂 Project Structure

main\
│── Attribution.md   #Contains attribution to libraries and tech-stacks used\
│── Presentation.pptx   #A presentation of our project\
│── README.md\
│── deepfake_detection_system_model.py       # CNN training code\
│── deepfake_detection_system_web_app.py             # Streamlit app for predictions\
│── requirements.txt   #A file of requirements\


## 🧠 Model Training (CNN)

1. Architecture:

- Input → Rescaling → Conv2D (32 filters) → MaxPooling

- Conv2D (64 filters) → MaxPooling

- Conv2D (128 filters) → MaxPooling

- Flatten → Dense(128, ReLU) → Dropout(0.5)

- Dense(1, Sigmoid) → Output


2. Compilation & Training:

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop])


3. Evaluation & Visualization:

- Accuracy/Loss plots

- Confusion Matrix (counts + percentages)

- Classification Report

## 🚀 Deployment (Streamlit App)

    @st.cache_resource
    def load_model_and_metadata():
        metadata = joblib.load("deepfake_detection_system_model.joblib")
        class_names = metadata["class_names"]
        model = load_model(metadata["model_path"])
        return model, class_names

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        prediction = model.predict(img_array)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
        st.write(f"**Class:** {predicted_class}, **Confidence:** {confidence*100:.2f}%")

## 🌟 Key Features

✅ CNN-based Deepfake detection\
✅ User-friendly UI\
✅ Confidence score included\
✅ Extendable as API\
✅ Improves with more data


## 🔮 Future Scope

- Real-time deepfake detection for live streaming

- Stronger AI models to handle evolving deepfakes

- Larger, diverse datasets

- Public awareness + media integration

- Legal/ethical frameworks for misuse prevention
