
import React from 'react';
import { Layers, MapPin } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import ValueSlider from '../ui/ValueSlider';
import FeatureCard from '../ui/FeatureCard';
import InfoTooltip from '../ui/InfoTooltip';

interface GeologicalFeaturesProps {
  lithology: string;
  setLithology: (value: string) => void;
  distanceToFaults: number;
  setDistanceToFaults: (value: number) => void;
}

const GeologicalFeatures: React.FC<GeologicalFeaturesProps> = ({
  lithology,
  setLithology,
  distanceToFaults,
  setDistanceToFaults
}) => {
  // Rock types with descriptions
  const rockTypes = [
    { id: 'granite', name: 'Granite', description: 'Hard, crystalline rock. Generally stable but can be susceptible to weathering.' },
    { id: 'limestone', name: 'Limestone', description: 'Sedimentary rock that can dissolve in water, potentially creating unstable conditions.' },
    { id: 'sandstone', name: 'Sandstone', description: 'Sedimentary rock with varying degrees of permeability and strength.' },
    { id: 'shale', name: 'Shale', description: 'Fine-grained sedimentary rock that can be weak and prone to sliding when wet.' },
    { id: 'schist', name: 'Schist', description: 'Metamorphic rock with layered structure that can create plane of weakness.' },
    { id: 'basalt', name: 'Basalt', description: 'Igneous rock that is generally stable but can weather to clay minerals.' },
  ];

  const selectedRock = rockTypes.find(rock => rock.id === lithology);
  const faultInfo = "Proximity to geological faults can influence landslide susceptibility. Fault zones are often associated with fractured rock masses and can serve as pathways for water infiltration.";

  return (
    <FeatureCard
      title="Geological Features"
      icon={<Layers className="h-5 w-5" />}
      className="w-full"
    >
      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center">
            <label className="text-sm font-medium">Lithology (Rock Type)</label>
            <InfoTooltip 
              content="The type of bedrock affects slope stability, drainage patterns, and soil development."
            />
          </div>
          
          <Select value={lithology} onValueChange={setLithology}>
            <SelectTrigger>
              <SelectValue placeholder="Select rock type" />
            </SelectTrigger>
            <SelectContent>
              {rockTypes.map(rock => (
                <SelectItem key={rock.id} value={rock.id}>
                  {rock.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          {selectedRock && (
            <div className="text-xs text-muted-foreground mt-1 p-2 bg-muted/40 rounded">
              {selectedRock.description}
            </div>
          )}
        </div>

        <div>
          <div className="flex items-center mb-2">
            <label className="text-sm font-medium">Distance to Fault Lines</label>
            <InfoTooltip content={faultInfo} />
          </div>
          
          <ValueSlider
            label="Distance"
            min={0}
            max={5000}
            step={100}
            defaultValue={distanceToFaults}
            unit="m"
            onChange={setDistanceToFaults}
          />
          
          <div className="relative h-4 w-full bg-gradient-to-r from-red-100 to-green-100 rounded mt-2">
            <MapPin 
              className="absolute text-primary h-5 w-5" 
              style={{ 
                left: `calc(${(distanceToFaults / 5000) * 100}% - 10px)`,
                top: '-10px'
              }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Near fault (higher risk)</span>
            <span>Far from fault (lower risk)</span>
          </div>
        </div>
      </div>
    </FeatureCard>
  );
};

export default GeologicalFeatures;
