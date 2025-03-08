
import React from 'react';
import { CloudRain, Trees, Droplets } from 'lucide-react';
import ValueSlider from '../ui/ValueSlider';
import FeatureCard from '../ui/FeatureCard';
import InfoTooltip from '../ui/InfoTooltip';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface EnvironmentalFeaturesProps {
  rainfall: number;
  setRainfall: (value: number) => void;
  vegetationDensity: number;
  setVegetationDensity: (value: number) => void;
  soilMoisture: number;
  setSoilMoisture: (value: number) => void;
  landUse: string;
  setLandUse: (value: string) => void;
}

const EnvironmentalFeatures: React.FC<EnvironmentalFeaturesProps> = ({
  rainfall,
  setRainfall,
  vegetationDensity,
  setVegetationDensity,
  soilMoisture,
  setSoilMoisture,
  landUse,
  setLandUse
}) => {
  const rainInfo = "Rainfall is a primary trigger for landslides. Heavy rainfall can saturate soil, increase its weight, and reduce cohesion between particles.";
  const vegetationInfo = "Vegetation helps stabilize slopes through root reinforcement and by reducing water content through transpiration.";
  const soilInfo = "Soil moisture affects the weight and cohesion of soil. Saturated soils have increased weight and reduced strength.";
  
  const landUseOptions = [
    { id: 'forest', name: 'Forest', risk: 'Low' },
    { id: 'agriculture', name: 'Agriculture', risk: 'Medium' },
    { id: 'urban', name: 'Urban/Developed', risk: 'Medium-High' },
    { id: 'mining', name: 'Mining/Excavation', risk: 'High' },
    { id: 'barren', name: 'Barren Land', risk: 'Variable' },
    { id: 'grassland', name: 'Grassland', risk: 'Medium-Low' },
  ];

  return (
    <FeatureCard
      title="Environmental Factors"
      icon={<CloudRain className="h-5 w-5" />}
      className="w-full"
    >
      <div className="space-y-5">
        <div>
          <div className="flex items-center mb-1">
            <span className="text-sm font-medium">Rainfall Intensity</span>
            <InfoTooltip content={rainInfo} />
          </div>
          
          <ValueSlider
            label="Daily Rainfall"
            min={0}
            max={300}
            step={5}
            defaultValue={rainfall}
            unit="mm"
            onChange={setRainfall}
          />
          
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Low</span>
            <span>Moderate</span>
            <span>Heavy</span>
            <span>Extreme</span>
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-5">
          <div className="flex-1">
            <div className="flex items-center mb-1">
              <span className="text-sm font-medium">Vegetation Density</span>
              <InfoTooltip content={vegetationInfo} />
            </div>
            
            <div className="relative">
              <ValueSlider
                label="Density"
                min={0}
                max={100}
                step={5}
                defaultValue={vegetationDensity}
                unit="%"
                onChange={setVegetationDensity}
              />
              
              <div className="absolute right-0 top-0 -mt-1">
                <div className="flex items-center space-x-1">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Trees
                      key={i}
                      className={`h-4 w-4 ${
                        i < Math.floor(vegetationDensity / 20)
                          ? 'text-secondary'
                          : 'text-muted-foreground/20'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex-1">
            <div className="flex items-center mb-1">
              <span className="text-sm font-medium">Soil Moisture</span>
              <InfoTooltip content={soilInfo} />
            </div>
            
            <ValueSlider
              label="Moisture"
              min={0}
              max={100}
              step={5}
              defaultValue={soilMoisture}
              unit="%"
              onChange={setSoilMoisture}
            />
            
            <div className="flex justify-between items-center mt-1">
              <span className="text-xs text-muted-foreground">Dry</span>
              <div className="flex-1 h-1.5 mx-2 rounded-full bg-gradient-to-r from-orange-100 via-blue-100 to-blue-300"></div>
              <span className="text-xs text-muted-foreground">Saturated</span>
            </div>
          </div>
        </div>
        
        <div>
          <div className="flex items-center mb-1">
            <span className="text-sm font-medium">Land Use</span>
            <InfoTooltip content="Land use affects slope stability through vegetation cover, water drainage, and human activities." />
          </div>
          
          <Select value={landUse} onValueChange={setLandUse}>
            <SelectTrigger>
              <SelectValue placeholder="Select land use" />
            </SelectTrigger>
            <SelectContent>
              {landUseOptions.map(option => (
                <SelectItem key={option.id} value={option.id}>
                  <div className="flex items-center justify-between w-full">
                    <span>{option.name}</span>
                    <span className={`text-xs ml-2 ${
                      option.risk === 'High' ? 'text-destructive' :
                      option.risk === 'Medium-High' ? 'text-amber-500' :
                      option.risk === 'Medium' ? 'text-amber-400' :
                      option.risk === 'Medium-Low' ? 'text-green-400' :
                      option.risk === 'Low' ? 'text-green-500' : 'text-muted-foreground'
                    }`}>
                      {option.risk} risk
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </FeatureCard>
  );
};

export default EnvironmentalFeatures;
