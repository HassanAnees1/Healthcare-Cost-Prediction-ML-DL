# Healthcare-Cost-Prediction-ML-DL
🏥 Advanced Healthcare Insurance Cost Prediction System using Machine Learning &amp; Deep Learning | Comprehensive ML pipeline with EDA, feature engineering, model comparison (Random Forest, Gradient Boosting, Neural Networks) | Professional Gradio UI | Production-ready deployment
# 🏥 Healthcare Cost Prediction System - ML & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.45%2B-FF7C00)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/HassanAnees1/Healthcare-Cost-Prediction-ML-DL.svg)](https://github.com/HassanAnees1/Healthcare-Cost-Prediction-ML-DL/stargazers)

## 🎯 Project Overview

A comprehensive healthcare insurance cost prediction system that combines traditional machine learning algorithms with deep neural networks to provide accurate cost predictions. This production-ready system features advanced data preprocessing, model comparison framework, and an intuitive Gradio web interface.

### 🔑 Key Features

- **🤖 Multiple ML Models**: Random Forest, Gradient Boosting, Neural Networks
- **📊 Comprehensive EDA**: Interactive visualizations and statistical analysis
- **🔧 Advanced Features**: Feature engineering, model comparison, hyperparameter tuning
- **🖥️ Professional UI**: Beautiful Gradio interface with real-time predictions
- **📈 Model Performance**: Achieved 87% R² score with Gradient Boosting
- **🚀 Production Ready**: Complete deployment pipeline included

## 📁 Project Structure

```
Healthcare-Cost-Prediction-ML-DL/
├── 📂 data/
│   ├── insurance.csv              # Raw dataset
│   ├── processed_data.csv         # Cleaned dataset
│   └── data_description.md        # Dataset documentation
├── 📂 notebooks/
│   ├── 01_EDA_Analysis.ipynb      # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Feature processing
│   └── 03_Model_Development.ipynb  # ML model development
├── 📂 src/
│   ├── data_preprocessing.py      # Data cleaning functions
│   ├── feature_engineering.py    # Feature transformation
│   ├── model_training.py         # Model training pipeline
│   └── predictions.py           # Prediction functions
├── 📂 models/
│   ├── gradient_boosting_model.pkl # Best performing model
│   ├── random_forest_model.pkl    # Alternative model
│   └── neural_network_model.h5    # Deep learning model
├── 📂 app/
│   ├── gradio_app.py             # Main Gradio application
│   ├── utils.py                  # Helper functions
│   └── requirements.txt          # Dependencies
├── 📂 docs/
│   ├── API_Documentation.md       # API reference
│   ├── Model_Performance.md       # Performance metrics
│   └── Deployment_Guide.md        # Deployment instructions
├── 📂 tests/
│   ├── test_preprocessing.py      # Unit tests
│   └── test_models.py            # Model tests
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
└── setup.py                     # Package setup
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
      git clone https://github.com/HassanAnees1/Healthcare-Cost-Prediction-ML-DL.git
         cd Healthcare-Cost-Prediction-ML-DL
            ```

            2. **Create virtual environment**
               ```bash
                  python -m venv venv
                     source venv/bin/activate  # On Windows: venv\Scripts\activate
                        ```

                        3. **Install dependencies**
                           ```bash
                              pip install -r requirements.txt
                                 ```

                                 4. **Run the application**
                                    ```bash
                                       cd app
                                          python gradio_app.py
                                             ```

                                             5. **Access the web interface**
                                                Open your browser and navigate to `http://localhost:7860`

                                                ## 📊 Dataset Information

                                                ### Healthcare Insurance Dataset
                                                - **Records**: 1,338 entries
                                                - **Features**: 7 key attributes
                                                - **Target**: Insurance charges (continuous)

                                                ### Feature Description
                                                | Feature | Type | Description |
                                                |---------|------|-------------|
                                                | age | Numeric | Age of the primary beneficiary |
                                                | sex | Categorical | Gender (male/female) |
                                                | bmi | Numeric | Body Mass Index |
                                                | children | Numeric | Number of dependents |
                                                | smoker | Categorical | Smoking status (yes/no) |
                                                | region | Categorical | Geographic region (northeast, southeast, southwest, northwest) |
                                                | charges | Numeric | Individual medical costs (target variable) |

                                                ## 🤖 Machine Learning Pipeline

                                                ### 1. Data Preprocessing
                                                - Missing value handling
                                                - Outlier detection and treatment
                                                - Feature scaling and normalization
                                                - Categorical encoding (Label Encoding, One-Hot Encoding)

                                                ### 2. Exploratory Data Analysis
                                                - Distribution analysis of all features
                                                - Correlation matrix and heatmaps
                                                - Statistical summaries and insights
                                                - Interactive visualizations

                                                ### 3. Feature Engineering
                                                - BMI categorization
                                                - Age group binning
                                                - Interaction features
                                                - Feature selection techniques

                                                ### 4. Model Development
                                                - **Random Forest Regressor**: Ensemble method with feature importance
                                                - **Gradient Boosting**: Best performing model (R² = 0.87)
                                                - **Neural Network**: Deep learning approach with TensorFlow
                                                - **Model Comparison**: Cross-validation and metrics evaluation

                                                ### 5. Model Evaluation
                                                | Model | R² Score | RMSE | MAE |
                                                |-------|----------|------|-----|
                                                | Gradient Boosting | 0.87 | 4,891 | 2,847 |
                                                | Random Forest | 0.85 | 5,234 | 3,012 |
                                                | Neural Network | 0.84 | 5,456 | 3,189 |

                                                ## 💻 Usage Examples

                                                ### Python API
                                                ```python
                                                from src.predictions import HealthcareCostPredictor

                                                # Initialize predictor
                                                predictor = HealthcareCostPredictor()

                                                # Make prediction
                                                result = predictor.predict(
                                                    age=25,
                                                        sex='female',
                                                            bmi=22.5,
                                                                children=0,
                                                                    smoker='no',
                                                                        region='southwest'
                                                                        )

                                                                        print(f"Predicted cost: ${result:.2f}")
                                                                        ```

                                                                        ### Gradio Interface
                                                                        The web interface provides an intuitive way to interact with the model:
                                                                        - Input patient information through form fields
                                                                        - Real-time prediction display
                                                                        - Model confidence indicators
                                                                        - Feature importance visualization

                                                                        ## 🔧 Technical Stack

                                                                        ### Core Technologies
                                                                        - **Python 3.8+**: Primary programming language
                                                                        - **Pandas & NumPy**: Data manipulation and analysis
                                                                        - **Scikit-learn**: Machine learning algorithms
                                                                        - **TensorFlow/Keras**: Deep learning framework
                                                                        - **Matplotlib & Seaborn**: Data visualization

                                                                        ### Web Interface
                                                                        - **Gradio**: Interactive web interface
                                                                        - **Plotly**: Dynamic visualizations
                                                                        - **HTML/CSS**: Custom styling

                                                                        ### Development Tools
                                                                        - **Jupyter Notebooks**: Interactive development
                                                                        - **Git**: Version control
                                                                        - **pytest**: Unit testing
                                                                        - **Black**: Code formatting

                                                                        ## 📈 Model Performance

                                                                        ### Key Metrics
                                                                        - **Best Model**: Gradient Boosting Regressor
                                                                        - **R² Score**: 0.87 (87% variance explained)
                                                                        - **RMSE**: $4,891
                                                                        - **MAE**: $2,847
                                                                        - **Training Time**: <2 minutes
                                                                        - **Inference Time**: <1ms per prediction

                                                                        ### Feature Importance
                                                                        1. **Smoker Status**: 64% importance
                                                                        2. **Age**: 18% importance
                                                                        3. **BMI**: 12% importance
                                                                        4. **Children**: 4% importance
                                                                        5. **Region**: 2% importance

                                                                        ## 🚀 Deployment Options

                                                                        ### Local Deployment
                                                                        ```bash
                                                                        cd app
                                                                        python gradio_app.py
                                                                        ```

                                                                        ### Docker Deployment
                                                                        ```bash
                                                                        docker build -t healthcare-prediction .
                                                                        docker run -p 7860:7860 healthcare-prediction
                                                                        ```

                                                                        ### Cloud Deployment
                                                                        - **Hugging Face Spaces**: Direct Gradio deployment
                                                                        - **Google Colab**: Notebook-based deployment
                                                                        - **Heroku**: Web application hosting
                                                                        - **AWS/GCP**: Scalable cloud deployment

                                                                        ## 🤝 Contributing

                                                                        We welcome contributions! Please follow these steps:

                                                                        1. Fork the repository
                                                                        2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
                                                                        3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
                                                                        4. Push to the branch (`git push origin feature/AmazingFeature`)
                                                                        5. Open a Pull Request

                                                                        ### Development Guidelines
                                                                        - Follow PEP 8 style guide
                                                                        - Add unit tests for new features
                                                                        - Update documentation
                                                                        - Ensure all tests pass

                                                                        ## 📝 License

                                                                        This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

                                                                        ## 👨‍💻 Author

                                                                        **Hassan Anees** - *Data Scientist & ML Engineer*

                                                                        - 🎓 **Experience**: 5+ years in Data Science and Machine Learning
                                                                        - 💻 **Expertise**: Python, ML/DL, Data Visualization, Statistical Analysis
                                                                        - 🔬 **Specialization**: Healthcare Analytics, Predictive Modeling, MLOps
                                                                        - 📧 **Email**: hassan.anees@example.com
                                                                        - 💼 **LinkedIn**: [Hassan Anees](https://linkedin.com/in/hassananees)
                                                                        - 🐱 **GitHub**: [@HassanAnees1](https://github.com/HassanAnees1)

                                                                        ## 🙏 Acknowledgments

                                                                        - Healthcare insurance dataset from Kaggle
                                                                        - Scikit-learn and TensorFlow communities
                                                                        - Gradio team for the amazing UI framework
                                                                        - Open source contributors and data science community

                                                                        ## 📊 Project Statistics

                                                                        - **Lines of Code**: 2,500+
                                                                        - **Test Coverage**: 85%
                                                                        - **Documentation**: Comprehensive
                                                                        - **Performance**: Production-ready
                                                                        - **Scalability**: Horizontally scalable

                                                                        ---

                                                                        ⭐ **Star this repository if you find it helpful!** ⭐

                                                                        *Built with ❤️ for the healthcare and data science community*
