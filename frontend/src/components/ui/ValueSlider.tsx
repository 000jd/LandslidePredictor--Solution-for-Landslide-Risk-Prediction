
import React, { useState, useEffect } from 'react';
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import InfoTooltip from './InfoTooltip';

interface ValueSliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  defaultValue: number;
  unit?: string;
  tooltip?: string;
  onChange: (value: number) => void;
}

const ValueSlider: React.FC<ValueSliderProps> = ({
  label,
  min,
  max,
  step = 1,
  defaultValue,
  unit = '',
  tooltip,
  onChange
}) => {
  const [value, setValue] = useState<number>(defaultValue);

  useEffect(() => {
    setValue(defaultValue);
  }, [defaultValue]);

  const handleSliderChange = (val: number[]) => {
    setValue(val[0]);
    onChange(val[0]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue)) {
      const boundedValue = Math.min(Math.max(newValue, min), max);
      setValue(boundedValue);
      onChange(boundedValue);
    }
  };

  return (
    <div className="space-y-3 w-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <label className="text-sm font-medium">{label}</label>
          {tooltip && <InfoTooltip content={tooltip} />}
        </div>
        <div className="flex items-center space-x-2">
          <Input
            type="number"
            value={value}
            onChange={handleInputChange}
            className="w-16 h-8 text-right"
            min={min}
            max={max}
            step={step}
          />
          <span className="text-sm text-muted-foreground w-6">{unit}</span>
        </div>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={handleSliderChange}
      />
    </div>
  );
};

export default ValueSlider;
