import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from '@/hooks/use-toast';
import PageTransition from '@/components/animations/PageTransition';
import FadeIn from '@/components/animations/FadeIn';
import ValueSlider from '@/components/ui/ValueSlider';
import CircularSlider from '@/components/ui/CircularSlider';
import GeographicalFeatures from '@/components/features/GeographicalFeatures';
import { fetchAvailableModels, predictLandslideRisk, Model } from '@/services/apiService';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from '@/components/ui/label';
import InfoTooltip from '@/components/ui/InfoTooltip';
import { TooltipProvider } from '@radix-ui/react-tooltip';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const Predict = () => {
  const navigate = useNavigate();
  
  // Geographical features state
  const [elevation, setElevation] = useState(500);
  const [slope, setSlope] = useState(30);
  const [aspect, setAspect] = useState(180);
  
  // Environmental features state
  const [rainfallDaily, setRainfallDaily] = useState(30);
  const [rainfallMonthly, setRainfallMonthly] = useState(170);
  const [vegetationDensity, setVegetationDensity] = useState(0.3);
  const [soilMoisture, setSoilMoisture] = useState(50);
  const [soilDepth, setSoilDepth] = useState(2.0);
  const [snowMelt, setSnowMelt] = useState(0.0);
  const [landslideProb, setLandslideProb] = useState(0.1);
  
  // Geological features state
  const [lithology, setLithology] = useState('granite');
  const [distanceToFaults, setDistanceToFaults] = useState(500);
  const [earthquakeMagnitude, setEarthquakeMagnitude] = useState(0.0);
  const [previousLandslides, setPreviousLandslides] = useState(0);
  
  // Human factors
  const [landUse, setLandUse] = useState('urban');
  const [humanActivity, setHumanActivity] = useState('low');

  // Model selection state
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  
  // Loading state
  const [isLoading, setIsLoading] = useState(false);

  // Fetch available models on component mount
  useEffect(() => {
    const getModels = async () => {
      setIsLoadingModels(true);
      try {
        const models = await fetchAvailableModels();
        console.log("Models fetched:", models);
        setAvailableModels(models);
        
        // Automatically select the first model in the list as default
        if (models.length > 0) {
          setSelectedModel(models[0].name);
          toast({
            title: "Model selected",
            description: `Using ${models[0].name} as the default model`,
            duration: 3000,
          });
        }
      } catch (error) {
        toast({
          title: "Failed to fetch models",
          description: "Using fallback models instead.",
          variant: "destructive",
          duration: 3000,
        });
        console.error("Error fetching models:", error);
      } finally {
        setIsLoadingModels(false);
      }
    };
    
    getModels();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    if (!selectedModel) {
      toast({
        title: "No Model Selected",
        description: "Please select a prediction model before continuing.",
        variant: "destructive",
        duration: 3000,
      });
      setIsLoading(false);
      return;
    }
    
    toast({
      title: "Processing prediction",
      description: "Sending data to the server...",
      duration: 2000,
    });

    try {
      const requestData = {
        model_name: selectedModel,
        elevation,
        slope,
        aspect,
        rainfall_daily: rainfallDaily,
        rainfall_monthly: rainfallMonthly,
        distance_to_faults: distanceToFaults,
        soil_depth: soilDepth,
        lithology,
        vegetation_density: vegetationDensity,
        land_use: landUse,
        earthquake_magnitude: earthquakeMagnitude,
        soil_moisture: soilMoisture,
        previous_landslides: previousLandslides,
        snow_melt: snowMelt,
        landslide_probability: landslideProb,
        human_activity: humanActivity
      };

      const response = await predictLandslideRisk(requestData);
      
      navigate('/results', { 
        state: {
          ...requestData,
          riskScore: response.risk_score
        }
      });
    } catch (error) {
      toast({
        title: "Prediction Failed",
        description: "There was an error connecting to the prediction service.",
        variant: "destructive",
        duration: 3000,
      });
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  // Model selection dialog component
  const ModelSelectionDialog = () => (
    <Dialog>
      <DialogTrigger asChild>
        <Button 
          variant="outline" 
          size="icon"
          className="fixed top-24 right-6 z-10 rounded-full shadow-md border-primary/20 bg-background hover:bg-accent"
          title="Select Prediction Model"
        >
          <Monitor className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Select Prediction Model</DialogTitle>
          <DialogDescription>
            Choose the machine learning model to use for landslide risk prediction.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          {isLoadingModels ? (
            <div className="flex justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            </div>
          ) : availableModels.length === 0 ? (
            <div className="text-center text-muted-foreground py-4">
              No models available. Please check the API connection.
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="model-select">Available Models</Label>
              <Select value={selectedModel || ''} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name} {model.description ? `- ${model.description}` : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          
          <div className="flex justify-between pt-2">
            <span className="text-sm text-muted-foreground">
              {selectedModel ? `Using: ${selectedModel}` : 'No model selected'}
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );

  return (
    <TooltipProvider>
      <PageTransition>
        <div className="min-h-screen pt-8 pb-16 px-6 relative">
          {/* Model Selection Dialog */}
          <ModelSelectionDialog />
          
          {/* Mesh Background */}
          <div className="absolute inset-0 -z-10 overflow-hidden">
            <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
            <div className="absolute inset-0 bg-gradient-to-b from-background to-background/50"></div>
          </div>
          
          <div className="max-w-6xl mx-auto">
            <FadeIn className="text-center mb-8">
              <h1 className="text-4xl font-light">Landslide Risk Prediction</h1>
              <p className="mt-2 text-muted-foreground max-w-2xl mx-auto">
                Enter geographical, geological, and environmental parameters to generate a landslide risk assessment
              </p>
              {selectedModel && (
                <p className="mt-1 text-sm text-primary">
                  Using model: <span className="font-semibold">{selectedModel}</span>
                </p>
              )}
            </FadeIn>

            <form onSubmit={handleSubmit} className="mt-8 space-y-8">
              {/* Geographical Features Section */}
              <FadeIn delay={0.1}>
                <GeographicalFeatures 
                  elevation={elevation}
                  setElevation={setElevation}
                  slope={slope}
                  setSlope={setSlope}
                  aspect={aspect}
                  setAspect={setAspect}
                />
              </FadeIn>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Environmental Factors */}
                <FadeIn delay={0.2} className="glass-card rounded-xl p-6">
                  <h2 className="text-xl font-medium mb-4">Environmental Factors</h2>
                  <div className="space-y-6">
                    <ValueSlider
                      label="Daily Rainfall"
                      min={0}
                      max={150}
                      step={0.1}
                      defaultValue={rainfallDaily}
                      unit="mm"
                      tooltip="Daily rainfall directly affects soil saturation and pore water pressure, increasing landslide risk."
                      onChange={setRainfallDaily}
                    />
                    
                    <ValueSlider
                      label="Monthly Rainfall"
                      min={0}
                      max={1000}
                      step={1}
                      defaultValue={rainfallMonthly}
                      unit="mm"
                      tooltip="Cumulative monthly rainfall is a key indicator of long-term soil moisture conditions."
                      onChange={setRainfallMonthly}
                    />
                    
                    <ValueSlider
                      label="Vegetation Density"
                      min={0}
                      max={1}
                      step={0.01}
                      defaultValue={vegetationDensity}
                      unit=""
                      tooltip="Higher vegetation density typically increases slope stability through root systems."
                      onChange={setVegetationDensity}
                    />
                    
                    <ValueSlider
                      label="Soil Moisture"
                      min={0}
                      max={100}
                      step={1}
                      defaultValue={soilMoisture}
                      unit="%"
                      tooltip="Soil moisture content affects shear strength and stability of slopes."
                      onChange={setSoilMoisture}
                    />
                  </div>
                </FadeIn>

                {/* Geological Features */}
                <FadeIn delay={0.3} className="glass-card rounded-xl p-6">
                  <h2 className="text-xl font-medium mb-4">Geological Features</h2>
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <Label htmlFor="lithology" className="text-sm font-medium">Lithology</Label>
                        <InfoTooltip content="Rock type affects resistance to weathering and potential for landslides." />
                      </div>
                      <Select value={lithology} onValueChange={setLithology}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select rock type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="granite">Granite</SelectItem>
                          <SelectItem value="limestone">Limestone</SelectItem>
                          <SelectItem value="sandstone">Sandstone</SelectItem>
                          <SelectItem value="shale">Shale</SelectItem>
                          <SelectItem value="basalt">Basalt</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <ValueSlider
                      label="Distance to Faults"
                      min={0}
                      max={5000}
                      step={10}
                      defaultValue={distanceToFaults}
                      unit="m"
                      tooltip="Proximity to fault lines increases seismic vulnerability and landslide risk."
                      onChange={setDistanceToFaults}
                    />
                    
                    <ValueSlider
                      label="Earthquake Magnitude"
                      min={0}
                      max={9}
                      step={0.1}
                      defaultValue={earthquakeMagnitude}
                      unit=""
                      tooltip="Recent seismic activity magnitude can trigger landslides (0 = no recent activity)."
                      onChange={setEarthquakeMagnitude}
                    />
                    
                    <ValueSlider
                      label="Previous Landslides"
                      min={0}
                      max={10}
                      step={1}
                      defaultValue={previousLandslides}
                      unit=""
                      tooltip="Number of historical landslide events in the area."
                      onChange={setPreviousLandslides}
                    />
                  </div>
                </FadeIn>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Additional Environmental Factors */}
                <FadeIn delay={0.4} className="glass-card rounded-xl p-6">
                  <h2 className="text-xl font-medium mb-4">Additional Factors</h2>
                  <div className="space-y-6">
                    <ValueSlider
                      label="Soil Depth"
                      min={0}
                      max={10}
                      step={0.1}
                      defaultValue={soilDepth}
                      unit="m"
                      tooltip="The depth of soil affects stability and landslide volume potential."
                      onChange={setSoilDepth}
                    />
                    
                    <ValueSlider
                      label="Snow Melt"
                      min={0}
                      max={50}
                      step={0.1}
                      defaultValue={snowMelt}
                      unit="mm/day"
                      tooltip="Rate of snow melt contributes to soil saturation and can trigger landslides."
                      onChange={setSnowMelt}
                    />
                    
                    <ValueSlider
                      label="Historical Landslide Probability"
                      min={0}
                      max={1}
                      step={0.01}
                      defaultValue={landslideProb}
                      unit=""
                      tooltip="Based on historical data, the general probability of landslides in this region."
                      onChange={setLandslideProb}
                    />
                  </div>
                </FadeIn>

                {/* Human Factors */}
                <FadeIn delay={0.5} className="glass-card rounded-xl p-6">
                  <h2 className="text-xl font-medium mb-4">Human Factors</h2>
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <Label htmlFor="landUse" className="text-sm font-medium">Land Use</Label>
                        <InfoTooltip content="Different land use types affect surface runoff, infiltration, and slope stability." />
                      </div>
                      <Select value={landUse} onValueChange={setLandUse}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select land use" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="urban">Urban</SelectItem>
                          <SelectItem value="agricultural">Agricultural</SelectItem>
                          <SelectItem value="forest">Forest</SelectItem>
                          <SelectItem value="barren">Barren</SelectItem>
                          <SelectItem value="wetland">Wetland</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <Label htmlFor="humanActivity" className="text-sm font-medium">Human Activity</Label>
                        <InfoTooltip content="Human activities like construction, deforestation, and mining can destabilize slopes." />
                      </div>
                      <Select value={humanActivity} onValueChange={setHumanActivity}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select activity level" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </FadeIn>
              </div>

              <FadeIn delay={0.6} className="flex justify-center">
                <Button 
                  type="submit" 
                  size="lg" 
                  className="px-8 rounded-full"
                  disabled={isLoading || !selectedModel}
                >
                  Generate Prediction <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </FadeIn>
            </form>
          </div>
        </div>
      </PageTransition>
    </TooltipProvider>
  );
};

export default Predict;
