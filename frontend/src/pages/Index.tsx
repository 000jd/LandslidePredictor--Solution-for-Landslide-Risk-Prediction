
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';
import { toast } from '@/components/ui/use-toast';
import FadeIn from '@/components/animations/FadeIn';
import PageTransition from '@/components/animations/PageTransition';
import ModelSelector from '@/components/ModelSelector';
import GeographicalFeatures from '@/components/features/GeographicalFeatures';
import GeologicalFeatures from '@/components/features/GeologicalFeatures';
import EnvironmentalFeatures from '@/components/features/EnvironmentalFeatures';
import { useTheme } from '@/components/theme-provider';
import { Moon, Sun } from 'lucide-react';

const Index = () => {
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();
  
  // Model selection state
  const [selectedModel, setSelectedModel] = useState('gradient-boost');
  
  // Geographical features state
  const [elevation, setElevation] = useState(800);
  const [slope, setSlope] = useState(25);
  const [aspect, setAspect] = useState(180);
  
  // Geological features state
  const [lithology, setLithology] = useState('sandstone');
  const [distanceToFaults, setDistanceToFaults] = useState(1000);
  
  // Environmental features state
  const [rainfall, setRainfall] = useState(50);
  const [vegetationDensity, setVegetationDensity] = useState(60);
  const [soilMoisture, setSoilMoisture] = useState(30);
  const [landUse, setLandUse] = useState('forest');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // In a real app, this would call your FastAPI backend
    // For now, we'll simulate a successful prediction

    toast({
      title: "Processing prediction",
      description: "Analyzing input parameters...",
      duration: 2000,
    });

    // Simulate API call delay
    setTimeout(() => {
      navigate('/results', { 
        state: {
          model: selectedModel,
          elevation,
          slope,
          aspect,
          lithology,
          distanceToFaults,
          rainfall,
          vegetationDensity,
          soilMoisture,
          landUse
        }
      });
    }, 1500);
  };

  return (
    <PageTransition>
      <div className="min-h-screen relative">
        {/* Mesh Background */}
        <div className="absolute inset-0 -z-10 opacity-70 dark:opacity-30">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#80b3ff44_1px,transparent_1px),linear-gradient(to_bottom,#80b3ff44_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
          <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 to-indigo-50/50 dark:from-blue-950/20 dark:to-indigo-950/20" />
        </div>
        
        {/* Theme Toggle */}
        <div className="absolute top-6 right-6 z-10">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-full"
          >
            {theme === "dark" ? (
              <Sun className="h-[1.2rem] w-[1.2rem]" />
            ) : (
              <Moon className="h-[1.2rem] w-[1.2rem]" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>

        {/* Main Content */}
        <div className="pt-12 px-6 max-w-6xl mx-auto">
          <FadeIn className="text-center mb-8">
            <div className="inline-block bg-primary/10 text-primary rounded-full px-4 py-1.5 text-sm font-medium mb-2">
              Advanced Landslide Risk Assessment
            </div>
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-light tracking-tight leading-tight md:leading-tight">
              Predict landslide risks with <span className="font-normal text-primary">precision</span>
            </h1>
            <p className="max-w-2xl mx-auto text-lg text-muted-foreground">
              Our AI-powered tool helps you assess landslide risk factors through an intuitive interface.
            </p>
          </FadeIn>

          {/* Prediction Form Section */}
          <section className="py-8">
            <form onSubmit={handleSubmit} className="mt-6">
              <div className="flex flex-col lg:flex-row gap-6 mb-6">
                <div className="w-full lg:w-1/3">
                  <FadeIn delay={0.1}>
                    <ModelSelector
                      selectedModel={selectedModel}
                      onSelectModel={setSelectedModel}
                    />
                  </FadeIn>
                </div>
                
                <div className="w-full lg:w-2/3">
                  <FadeIn delay={0.2}>
                    <GeographicalFeatures
                      elevation={elevation}
                      setElevation={setElevation}
                      slope={slope}
                      setSlope={setSlope}
                      aspect={aspect}
                      setAspect={setAspect}
                    />
                  </FadeIn>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6 mb-8">
                <FadeIn delay={0.3}>
                  <GeologicalFeatures
                    lithology={lithology}
                    setLithology={setLithology}
                    distanceToFaults={distanceToFaults}
                    setDistanceToFaults={setDistanceToFaults}
                  />
                </FadeIn>
                
                <FadeIn delay={0.4}>
                  <EnvironmentalFeatures
                    rainfall={rainfall}
                    setRainfall={setRainfall}
                    vegetationDensity={vegetationDensity}
                    setVegetationDensity={setVegetationDensity}
                    soilMoisture={soilMoisture}
                    setSoilMoisture={setSoilMoisture}
                    landUse={landUse}
                    setLandUse={setLandUse}
                  />
                </FadeIn>
              </div>

              <FadeIn delay={0.5} className="flex justify-center mb-12">
                <Button 
                  type="submit" 
                  size="lg" 
                  className="px-8 rounded-full"
                >
                  Generate Prediction <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </FadeIn>
            </form>
          </section>
        </div>
      </div>
    </PageTransition>
  );
};

export default Index;
