
export interface PredictionRequest {
  model_name?: string;
  elevation: number;
  slope: number;
  aspect: number;
  rainfall_daily: number;
  rainfall_monthly: number;
  distance_to_faults: number;
  soil_depth: number;
  lithology: string;
  vegetation_density: number;
  land_use: string;
  earthquake_magnitude: number;
  soil_moisture: number;
  previous_landslides: number;
  snow_melt: number;
  landslide_probability: number;
  human_activity: string;
}

export interface PredictionResponse {
  risk_score: number;
  message: string;
}

export interface Model {
  name: string;
  description?: string;
  created_at?: string;
}

// Fallback models in case the API is not available
const fallbackModels: Model[] = [
  { name: "xgboost_20250306_193033", description: "XGBoost Model (Default)" },
  { name: "random_forest_20250305_142211", description: "Random Forest Model" },
  { name: "gradient_boost_20250307_083022", description: "Gradient Boosting Model" }
];

export const fetchAvailableModels = async (): Promise<Model[]> => {
  try {
    const response = await fetch('http://localhost:8000/api/v1/models', {
      // Add a short timeout to avoid long waiting times if the server is down
      signal: AbortSignal.timeout(3000)
    });
    
    if (!response.ok) {
      console.warn(`API returned error status: ${response.status}. Using fallback models.`);
      return fallbackModels;
    }
    
    const data = await response.json();
    console.log("Raw API response:", data);
    
    // Handle the case where the API returns an array of strings instead of Model objects
    if (Array.isArray(data) && data.length > 0) {
      if (typeof data[0] === 'string') {
        // Convert string array to Model array
        return data.map(modelName => ({
          name: modelName,
          description: modelName.includes('xgboost') ? 'XGBoost Model' :
                      modelName.includes('linear') ? 'Linear Regression Model' :
                      modelName.includes('lightgbm') ? 'Light GBM Model' :
                      modelName.includes('random') ? 'Random Forest Model' :
                      'Unknown Model Type'
        }));
      } else if (typeof data[0] === 'object' && data[0].name) {
        // Already in the correct format
        return data;
      }
    }
    
    console.warn("API returned unexpected format, using fallback models");
    return fallbackModels;
  } catch (error) {
    console.warn('Error fetching available models, using fallback models:', error);
    return fallbackModels;
  }
};

export const predictLandslideRisk = async (data: PredictionRequest): Promise<PredictionResponse> => {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    console.log("Prediction API response:", result);
    
    // Handle different response formats
    if (result.prediction !== undefined) {
      // New format: {"prediction": 0.5351960062980652}
      const riskScore = parseFloat(result.prediction);
      
      // Generate appropriate message based on risk score
      let message = "";
      if (riskScore < 0.33) {
        message = "Low risk of landslide based on the provided parameters.";
      } else if (riskScore < 0.66) {
        message = "Medium risk of landslide. Consider monitoring and preventive measures.";
      } else {
        message = "High risk of landslide. Immediate attention and mitigation advised.";
      }
      
      return {
        risk_score: riskScore,
        message: message
      };
    } else if (result.risk_score !== undefined) {
      // Original expected format
      return result;
    } else {
      // Unknown format, create a generic response
      console.warn("Unexpected prediction response format:", result);
      return {
        risk_score: 0.5, // Default to medium risk
        message: "Unable to parse prediction result. Please check parameters and try again."
      };
    }
  } catch (error) {
    console.error('Error predicting landslide risk:', error);
    
    // If we can't connect to the API, return a simulated response
    // This is just for demonstration purposes
    const simulatedRiskScore = Math.random() * 100;
    return {
      risk_score: simulatedRiskScore / 100, // Convert to 0-1 range
      message: "Simulated prediction (API unavailable): " + 
               (simulatedRiskScore > 70 ? "High risk" : 
                simulatedRiskScore > 30 ? "Medium risk" : "Low risk")
    };
  }
};
