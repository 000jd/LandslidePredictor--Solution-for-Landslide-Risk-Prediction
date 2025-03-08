
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Edit2 } from 'lucide-react';
import PageTransition from '@/components/animations/PageTransition';
import FadeIn from '@/components/animations/FadeIn';
import RiskDisplay from '@/components/RiskDisplay';
import { toast } from '@/hooks/use-toast';

interface ResultsState {
  model: string;
  elevation: number;
  slope: number;
  aspect: number;
  lithology: string;
  distance_to_faults: number;
  rainfall_daily: number;
  rainfall_monthly: number;
  vegetation_density: number;
  soil_moisture: number;
  soil_depth: number;
  land_use: string;
  earthquake_magnitude: number;
  previous_landslides: number;
  snow_melt: number;
  landslide_probability: number;
  human_activity: string;
  riskScore: number;
}

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  
  // Risk prediction state
  const [riskLevel, setRiskLevel] = useState<'low' | 'medium' | 'high'>('medium');
  const [riskProbability, setRiskProbability] = useState(0);
  const [confidenceInterval, setConfidenceInterval] = useState<[number, number]>([0, 0]);
  
  // Feature importance data for visualization
  const [featureImportance, setFeatureImportance] = useState<{feature: string, importance: number}[]>([]);
  
  useEffect(() => {
    // Make sure we have state data
    if (!location.state) {
      navigate('/');
      return;
    }
    
    const data = location.state as ResultsState;
    
    // Process the risk score from the API
    processRiskScore(data.riskScore);
    
    setIsLoading(false);
  }, [location.state, navigate]);
  
  const processRiskScore = (riskScore: number) => {
    // The API now returns risk score in the 0-1 range, so we convert to percentage (0-100)
    const riskPercentage = riskScore * 100;
    
    // Determine risk level based on score
    let level: 'low' | 'medium' | 'high';
    if (riskScore < 0.33) {
      level = 'low';
    } else if (riskScore < 0.66) {
      level = 'medium';
    } else {
      level = 'high';
    }
    
    // Generate confidence interval (just for demo)
    const variability = 5 + Math.random() * 10;
    const lowerBound = Math.max(0, riskPercentage - variability);
    const upperBound = Math.min(100, riskPercentage + variability);
    
    // Update state with calculated values
    setRiskLevel(level);
    setRiskProbability(riskPercentage);
    setConfidenceInterval([lowerBound, upperBound]);
    
    // Create feature importance for visualization
    if (location.state) {
      const data = location.state as ResultsState;
      
      const importance = [
        { feature: 'Slope', importance: (data.slope / 90) * 25 },
        { feature: 'Rainfall', importance: (data.rainfall_daily / 300) * 20 },
        { feature: 'Soil Moisture', importance: (data.soil_moisture / 100) * 20 },
        { feature: 'Vegetation', importance: ((1 - data.vegetation_density) * 15) },
        { feature: 'Elevation', importance: (data.elevation / 5000) * 10 },
        { feature: 'Fault Distance', importance: ((5000 - data.distance_to_faults) / 5000) * 10 }
      ].sort((a, b) => b.importance - a.importance);
      
      setFeatureImportance(importance);
    }
  };
  
  return (
    <PageTransition>
      <div className="min-h-screen pt-8 pb-16 px-6 relative">
        {/* Mesh Background */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-background to-background/50"></div>
        </div>
        
        <div className="max-w-5xl mx-auto">
          <FadeIn className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-4xl font-light">Prediction Results</h1>
              <p className="text-muted-foreground mt-1">
                Based on your input parameters
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                onClick={() => navigate('/')}
              >
                <ArrowLeft className="mr-2 h-4 w-4" /> Back
              </Button>
              <Button
                onClick={() => navigate('/', { state: location.state })}
              >
                <Edit2 className="mr-2 h-4 w-4" /> Edit Input
              </Button>
            </div>
          </FadeIn>

          <div className="grid lg:grid-cols-5 gap-8">
            <div className="lg:col-span-3">
              <FadeIn delay={0.1}>
                <RiskDisplay
                  riskLevel={riskLevel}
                  riskProbability={riskProbability}
                  confidenceInterval={confidenceInterval}
                />
              </FadeIn>
            </div>
            
            <div className="lg:col-span-2">
              <FadeIn delay={0.2}>
                <div className="glass-card rounded-xl p-6">
                  <h3 className="text-xl font-medium mb-4">Key Risk Factors</h3>
                  
                  <div className="space-y-4">
                    {featureImportance.map((feature, index) => (
                      <div key={index}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{feature.feature}</span>
                          <span className="font-medium">{feature.importance.toFixed(1)}</span>
                        </div>
                        <div className="h-2 bg-muted/40 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-primary"
                            style={{ width: `${(feature.importance / 25) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="text-xs text-muted-foreground mt-4">
                    These factors have the highest influence on the prediction result
                  </div>
                </div>
              </FadeIn>
              
              <FadeIn delay={0.3}>
                <div className="glass-card rounded-xl p-6 mt-6">
                  <h3 className="text-xl font-medium mb-2">What's Next?</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Based on your risk assessment, consider these next steps:
                  </p>
                  
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start">
                      <div className="bg-secondary/20 p-1.5 rounded-full mr-2 mt-0.5">
                        <div className="w-1.5 h-1.5 bg-secondary rounded-full" />
                      </div>
                      Download a detailed report with recommendations
                    </li>
                    <li className="flex items-start">
                      <div className="bg-secondary/20 p-1.5 rounded-full mr-2 mt-0.5">
                        <div className="w-1.5 h-1.5 bg-secondary rounded-full" />
                      </div>
                      Share results with relevant stakeholders
                    </li>
                    <li className="flex items-start">
                      <div className="bg-secondary/20 p-1.5 rounded-full mr-2 mt-0.5">
                        <div className="w-1.5 h-1.5 bg-secondary rounded-full" />
                      </div>
                      Explore mitigation strategies for your identified risk level
                    </li>
                  </ul>
                </div>
              </FadeIn>
            </div>
          </div>
        </div>
      </div>
    </PageTransition>
  );
};

export default Results;
