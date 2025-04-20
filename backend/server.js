const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");
const FormData = require("form-data");
const sharp = require("sharp");
const https = require("https");
const mongoose = require("mongoose");

// Load environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

// MongoDB Connection
mongoose
  .connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("MongoDB connected successfully"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Create schemas and models
const healthProfileSchema = new mongoose.Schema({
  userId: { type: String, required: true, unique: true },
  age: Number,
  gender: String,
  weight: Number,
  height: Number,
  healthConditions: String,
  dietaryPreferences: String,
  allergies: String,
  fitnessGoals: String,
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

const foodAnalysisSchema = new mongoose.Schema({
  userId: String,
  foodName: String,
  nutritionInfo: {
    calories: String,
    protein: String,
    carbohydrates: String,
    fat: String,
  },
  recommendations: String,
  imageUrl: String,
  createdAt: { type: Date, default: Date.now },
});

const HealthProfile = mongoose.model("HealthProfile", healthProfileSchema);
const FoodAnalysis = mongoose.model("FoodAnalysis", foodAnalysisSchema);

// Middleware
app.use(cors());
app.use(express.json());

// Set up multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage });

// Create a custom axios instance for Grok API with proper SSL configuration
const grokAxios = axios.create({
  baseURL: "https://api.grok.ai",
  timeout: 10000,
  httpsAgent: new https.Agent({
    rejectUnauthorized: true,
    servername: "api.grok.ai",
  }),
});

// Routes
app.post("/api/analyze-food", upload.single("foodImage"), async (req, res) => {
  let compressedImagePath = "";

  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided" });
    }

    console.log("File received:", req.file);

    // Create a compressed JPEG file
    compressedImagePath = path.join(
      __dirname,
      "uploads",
      "compressed-" + req.file.filename
    );

    // Process with Sharp - resize and convert to JPEG
    await sharp(req.file.path)
      .resize(600)
      .jpeg({ quality: 70 })
      .toFile(compressedImagePath);

    // Read the compressed file
    const imageBuffer = fs.readFileSync(compressedImagePath);

    // LogMeal API integration
    const logMealApiKey = process.env.LOGMEAL_API_KEY;
    if (!logMealApiKey) {
      return res.status(500).json({ error: "LogMeal API key is missing" });
    }

    const logMealUrl = "https://api.logmeal.es/v2/image/recognition/dish";

    // Create FormData and append image
    const formData = new FormData();
    formData.append("image", imageBuffer, {
      filename: "food-image.jpg",
      contentType: "image/jpeg",
    });

    console.log("Sending request to LogMeal API...");

    // Send request to LogMeal API
    const logMealResponse = await axios.post(logMealUrl, formData, {
      headers: {
        Authorization: `Bearer ${logMealApiKey}`,
        ...formData.getHeaders(),
      },
    });

    console.log("LogMeal API response received");

    // Process LogMeal response
    const foodIdentification = logMealResponse.data;

    // Extract food name from the recognition results
    let foodName = "Unknown Food";
    let foodId = null;
    let nutritionInfo = {
      name: "Unknown Food",
      nutrition: {
        calories: "N/A",
        protein: "N/A",
        carbohydrates: "N/A",
        fat: "N/A",
      },
    };

    if (
      foodIdentification.recognition_results &&
      foodIdentification.recognition_results.length > 0
    ) {
      foodId = foodIdentification.recognition_results[0].id;
      foodName = foodIdentification.recognition_results[0].name;

      console.log(`Identified food: ${foodName} (ID: ${foodId})`);
      nutritionInfo.name = foodName;

      // Try to get nutrition info from LogMeal API
      try {
        const nutritionUrl = `https://api.logmeal.es/v2/recipe/info/${foodId}`;
        console.log(`Fetching nutrition data from: ${nutritionUrl}`);

        const nutritionResponse = await axios.get(nutritionUrl, {
          headers: { Authorization: `Bearer ${logMealApiKey}` },
        });

        if (nutritionResponse.data && nutritionResponse.data.nutrition) {
          nutritionInfo = nutritionResponse.data;
          console.log("Nutrition data retrieved successfully");
        }
      } catch (nutritionError) {
        console.log(
          "Failed to get detailed nutrition, trying alternative endpoint"
        );

        try {
          const altNutritionUrl = `https://api.logmeal.es/v2/nutrition/recipe/${foodId}`;
          console.log(`Trying alternative endpoint: ${altNutritionUrl}`);

          const altNutritionResponse = await axios.get(altNutritionUrl, {
            headers: { Authorization: `Bearer ${logMealApiKey}` },
          });

          if (
            altNutritionResponse.data &&
            altNutritionResponse.data.nutrition
          ) {
            nutritionInfo = altNutritionResponse.data;
            console.log("Alternative nutrition data retrieved");
          }
        } catch (altError) {
          console.error("Both nutrition endpoints failed:", altError.message);

          // If LogMeal nutrition endpoints fail, use Grok API for estimation
          console.log("Using Grok API for nutrition estimation");

          try {
            const nutritionValues = await getGrokNutritionEstimate(foodName);
            nutritionInfo.nutrition = nutritionValues;
            console.log("Retrieved nutrition estimates from Grok");
          } catch (grokError) {
            console.error(
              "Failed to get nutrition from Grok:",
              grokError.message
            );
            // Keep default N/A values
          }
        }
      }
    } else {
      return res
        .status(400)
        .json({ error: "Could not identify the food in the image" });
    }

    // Get user profile
    const userId = req.body.userId || "default";
    let userProfile = {};

    try {
      const profile = await HealthProfile.findOne({ userId: userId });
      if (profile) {
        userProfile = profile;
      }
    } catch (profileError) {
      console.error("Error retrieving user profile:", profileError.message);
    }

    // Get recommendations from Grok API
    let recommendations =
      "Unable to retrieve personalized recommendations. Please try again later.";
    try {
      recommendations = await getGrokRecommendations(
        nutritionInfo,
        userProfile
      );
      console.log("Successfully retrieved recommendations from Grok");
    } catch (grokError) {
      console.error(
        "Failed to get recommendations from Grok:",
        grokError.message
      );
    }

    // Save the analysis to MongoDB
    try {
      await FoodAnalysis.create({
        userId: userId,
        foodName: nutritionInfo.name,
        nutritionInfo: {
          calories: nutritionInfo.nutrition?.calories || "N/A",
          protein: nutritionInfo.nutrition?.protein || "N/A",
          carbohydrates: nutritionInfo.nutrition?.carbohydrates || "N/A",
          fat: nutritionInfo.nutrition?.fat || "N/A",
        },
        recommendations: recommendations,
      });
      console.log("Food analysis saved to database");
    } catch (dbError) {
      console.error(
        "Failed to save food analysis to database:",
        dbError.message
      );
    }

    // Send combined response to client
    const result = {
      name: nutritionInfo.name,
      calories: nutritionInfo.nutrition?.calories || "N/A",
      protein: nutritionInfo.nutrition?.protein || "N/A",
      carbs: nutritionInfo.nutrition?.carbohydrates || "N/A",
      fat: nutritionInfo.nutrition?.fat || "N/A",
      recommendations: recommendations,
    };

    res.json(result);
  } catch (error) {
    console.error("Error analyzing food:", error.message);
    if (error.response) {
      console.error("API Response Error:", {
        status: error.response.status,
        data: error.response.data,
      });
    }
    res
      .status(500)
      .json({ error: "Failed to analyze food image", details: error.message });
  } finally {
    // Clean up uploaded files with retry mechanism
    setTimeout(() => {
      try {
        if (req.file && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
        if (compressedImagePath && fs.existsSync(compressedImagePath)) {
          fs.unlinkSync(compressedImagePath);
        }
      } catch (cleanupError) {
        console.error("Error cleaning up files:", cleanupError.message);
      }
    }, 1000);
  }
});

app.post("/api/save-health-profile", async (req, res) => {
  try {
    const profileData = req.body;
    const userId = req.body.userId || "default";

    // Find and update if exists, otherwise create new
    const result = await HealthProfile.findOneAndUpdate(
      { userId: userId },
      {
        ...profileData,
        updatedAt: Date.now(),
      },
      {
        new: true,
        upsert: true,
      }
    );

    res
      .status(200)
      .json({ message: "Health profile saved successfully", profile: result });
  } catch (error) {
    console.error("Error saving health profile:", error);
    res.status(500).json({ error: "Failed to save health profile" });
  }
});

app.get("/api/health-profile/:userId", async (req, res) => {
  try {
    const userId = req.params.userId || "default";
    const profile = await HealthProfile.findOne({ userId: userId });

    if (!profile) {
      return res.status(404).json({ message: "Profile not found" });
    }

    res.status(200).json(profile);
  } catch (error) {
    console.error("Error retrieving health profile:", error);
    res.status(500).json({ error: "Failed to retrieve health profile" });
  }
});

// Add endpoint to get user's food analysis history
app.get("/api/food-history/:userId", async (req, res) => {
  try {
    const userId = req.params.userId || "default";
    const history = await FoodAnalysis.find({ userId: userId })
      .sort({ createdAt: -1 })
      .limit(10);

    res.status(200).json(history);
  } catch (error) {
    console.error("Error retrieving food history:", error);
    res.status(500).json({ error: "Failed to retrieve food history" });
  }
});

/**
 * Get nutrition estimates from Grok API
 * @param {string} foodName - The name of the food
 * @returns {Promise<Object>} - Nutrition values
 */
async function getGrokNutritionEstimate(foodName) {
  try {
    const grokApiKey = process.env.GROK_API_KEY;

    // Check that we have the API key
    if (!grokApiKey) {
      console.error("Missing Grok API key");
      throw new Error("Missing Grok API key");
    }

    console.log("Sending nutrition request to Grok API");

    // Use the custom grokAxios instance with SSL configuration
    const response = await grokAxios({
      method: "post",
      url: "/v1/completions",
      headers: {
        Authorization: `Bearer ${grokApiKey}`,
        "Content-Type": "application/json",
      },
      data: {
        model: "grok-1",
        prompt: `You are a nutrition database. Based on nutritional science, provide the nutritional values for "${foodName}" per 100g serving. Respond in this exact JSON format and nothing else:
{
  "calories": "number in kcal",
  "protein": "number in g",
  "carbohydrates": "number in g",
  "fat": "number in g"
}`,
        max_tokens: 150,
        temperature: 0.7,
      },
    });

    console.log("Received response from Grok API for nutrition");

    if (
      response.data &&
      response.data.choices &&
      response.data.choices.length > 0
    ) {
      const responseText = response.data.choices[0].text.trim();

      // Try to parse the JSON response
      try {
        // Extract JSON object if it's wrapped in text
        const jsonMatch = responseText.match(/(\{[\s\S]*\})/);
        const jsonStr = jsonMatch ? jsonMatch[0] : responseText;
        const nutritionData = JSON.parse(jsonStr);

        return {
          calories: nutritionData.calories || "N/A",
          protein: nutritionData.protein || "N/A",
          carbohydrates: nutritionData.carbohydrates || "N/A",
          fat: nutritionData.fat || "N/A",
        };
      } catch (parseError) {
        console.error(
          "Error parsing Grok nutrition JSON response:",
          parseError.message
        );
        console.log("Raw response:", responseText);
        throw new Error("Failed to parse Grok nutrition response");
      }
    } else {
      console.error("Unexpected Grok API response format:", response.data);
      throw new Error("Unexpected Grok API response format");
    }
  } catch (error) {
    console.error("Error in getGrokNutritionEstimate:", error.message);
    throw error;
  }
}

/**
 * Get recommendations from Grok API
 * @param {Object} nutritionInfo - Nutrition information for the food
 * @param {Object} userProfile - User health profile
 * @returns {Promise<string>} - Personalized recommendations
 */
async function getGrokRecommendations(nutritionInfo, userProfile) {
  try {
    const grokApiKey = process.env.GROK_API_KEY;

    // Check that we have the API key
    if (!grokApiKey) {
      console.error("Missing Grok API key");
      throw new Error("Missing Grok API key");
    }

    console.log("Sending recommendations request to Grok API");

    // Use the custom grokAxios instance with SSL configuration
    const response = await grokAxios({
      method: "post",
      url: "/v1/completions",
      headers: {
        Authorization: `Bearer ${grokApiKey}`,
        "Content-Type": "application/json",
      },
      data: {
        model: "grok-1",
        prompt: `You are a professional nutritional advisor. Analyze this food and provide helpful health recommendations:
        
Food: ${nutritionInfo.name || "Unknown food"}

Nutritional information per serving:
- Calories: ${nutritionInfo.nutrition?.calories || "unknown"} kcal
- Protein: ${nutritionInfo.nutrition?.protein || "unknown"} g
- Carbohydrates: ${nutritionInfo.nutrition?.carbohydrates || "unknown"} g
- Fat: ${nutritionInfo.nutrition?.fat || "unknown"} g

User health profile:
- Age: ${userProfile.age || "unknown"}
- Gender: ${userProfile.gender || "unknown"}
- Weight: ${userProfile.weight || "unknown"} kg
- Height: ${userProfile.height || "unknown"} cm
- Health conditions: ${userProfile.healthConditions || "none specified"}
- Dietary preferences: ${userProfile.dietaryPreferences || "none specified"}
- Allergies: ${userProfile.allergies || "none specified"}
- Fitness goals: ${userProfile.fitnessGoals || "none specified"}

Provide 3-5 personalized recommendations about:
1. Nutritional benefits or concerns
2. Appropriate portion sizes
3. Healthy preparation methods
4. How this food fits into their overall diet
5. Potential alternatives if this food doesn't align with their health goals

Keep your advice concise (150 words max), practical, and specifically tailored to their profile.`,
        max_tokens: 300,
        temperature: 0.7,
      },
    });

    console.log("Received response from Grok API for recommendations");

    if (
      response.data &&
      response.data.choices &&
      response.data.choices.length > 0
    ) {
      return response.data.choices[0].text.trim();
    } else {
      console.error("Unexpected Grok API response format:", response.data);
      throw new Error("Unexpected Grok API response format");
    }
  } catch (error) {
    console.error("Error in getGrokRecommendations:", error.message);
    throw error;
  }
}

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
