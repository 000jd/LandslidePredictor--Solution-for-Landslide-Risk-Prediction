
import React from 'react';
import { Mountain, Compass, TrendingUp } from 'lucide-react';
import ValueSlider from '../ui/ValueSlider';
import FeatureCard from '../ui/FeatureCard';
import CircularSlider from '../ui/CircularSlider';
import InfoTooltip from '../ui/InfoTooltip';

interface GeographicalFeaturesProps {
  elevation: number;
  setElevation: (value: number) => void;
  slope: number;
  setSlope: (value: number) => void;
  aspect: number;
  setAspect: (value: number) => void;
}

const GeographicalFeatures: React.FC<GeographicalFeaturesProps> = ({
  elevation,
  setElevation,
  slope,
  setSlope,
  aspect,
  setAspect
}) => {
  // Sample explanations for tooltips
  const elevationInfo = "Elevation affects landslide risk by influencing rainfall, vegetation, and soil conditions. Higher elevations often experience more rainfall and have thinner soils, potentially increasing risk.";
  const slopeInfo = "Slope angle is one of the most critical factors in landslide prediction. Steeper slopes have higher potential energy, making them more susceptible to failure.";
  const aspectInfo = "Aspect (the direction a slope faces) influences landslide risk by affecting soil moisture through sun exposure, vegetation growth, and weather patterns.";

  return (
    <FeatureCard
      title="Geographical Features"
      icon={<Mountain className="h-5 w-5" />}
      className="w-full"
    >
      <div className="space-y-5">
        <div>
          <ValueSlider
            label="Elevation"
            tooltip={elevationInfo}
            min={0}
            max={5000}
            step={10}
            defaultValue={elevation}
            unit="m"
            onChange={setElevation}
          />
          
          <div className="h-2 w-full bg-muted/50 rounded-full mt-1">
            <div
              className="h-full bg-primary/20 rounded-full"
              style={{ width: `${(elevation / 5000) * 100}%` }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        <div>
          <ValueSlider
            label="Slope"
            tooltip={slopeInfo}
            min={0}
            max={90}
            step={1}
            defaultValue={slope}
            unit="°"
            onChange={setSlope}
          />
          
          <div className="h-6 w-full bg-gradient-to-r from-green-100 via-yellow-100 to-red-100 rounded mt-1 relative">
            <div
              className="absolute h-full w-0.5 bg-foreground"
              style={{ left: `${(slope / 90) * 100}%` }}
            />
            <div className="flex justify-between text-xs absolute inset-x-0 -bottom-5">
              <span>Flat</span>
              <span>Moderate</span>
              <span>Steep</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center mt-6">
          <div className="flex items-center mb-1">
            <span className="text-sm font-medium mr-1">Aspect</span>
            <InfoTooltip content={aspectInfo} />
          </div>
          <CircularSlider
            value={aspect}
            onChange={setAspect}
            size={200}
          />
        </div>
      </div>
    </FeatureCard>
  );
};

export default GeographicalFeatures;
