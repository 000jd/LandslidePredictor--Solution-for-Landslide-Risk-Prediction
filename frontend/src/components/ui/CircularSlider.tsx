
import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface CircularSliderProps {
  size?: number;
  min?: number;
  max?: number;
  value: number;
  onChange: (value: number) => void;
  className?: string;
  label?: string;
}

const CircularSlider: React.FC<CircularSliderProps> = ({
  size = 200,
  min = 0,
  max = 360,
  value,
  onChange,
  className,
  label
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  const handlePointerMove = (event: PointerEvent) => {
    if (!isDragging || !sliderRef.current) return;
    
    const rect = sliderRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    const x = event.clientX - centerX;
    const y = event.clientY - centerY;
    
    // Calculate angle in degrees
    let angle = Math.atan2(y, x) * (180 / Math.PI);
    
    // Convert to 0-360 range
    angle = (angle + 360) % 360;
    
    // Map to min-max range
    const normalizedValue = min + (angle / 360) * (max - min);
    
    onChange(Math.round(normalizedValue));
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    } else {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    }
    
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isDragging]);

  // Convert value to angle for visual representation
  const angle = ((value - min) / (max - min)) * 360;
  
  // Calculate handle position
  const radius = size / 2 - 15;
  const handleX = radius * Math.cos((angle - 90) * (Math.PI / 180));
  const handleY = radius * Math.sin((angle - 90) * (Math.PI / 180));

  // Cardinal directions
  const cardinalDirections = [
    { label: 'N', angle: 0 },
    { label: 'E', angle: 90 },
    { label: 'S', angle: 180 },
    { label: 'W', angle: 270 }
  ];

  return (
    <div className="flex flex-col items-center space-y-2">
      {label && <span className="text-sm font-medium">{label}</span>}
      <div 
        ref={sliderRef}
        className={cn("relative rounded-full border cursor-pointer transition-colors hover:border-primary", className)}
        style={{ 
          width: size, 
          height: size,
        }}
        onPointerDown={() => setIsDragging(true)}
      >
        {/* Tick marks */}
        {Array.from({ length: 12 }).map((_, i) => {
          const tickAngle = i * 30;
          const tickRadius = radius + 5;
          const tickX = tickRadius * Math.cos((tickAngle - 90) * (Math.PI / 180));
          const tickY = tickRadius * Math.sin((tickAngle - 90) * (Math.PI / 180));
          
          return (
            <div 
              key={i}
              className="absolute w-1 h-1 bg-muted-foreground/50 rounded-full"
              style={{
                left: size / 2 + tickX - 0.5,
                top: size / 2 + tickY - 0.5,
                transform: 'translate(-50%, -50%)'
              }}
            />
          );
        })}
        
        {/* Cardinal directions */}
        {cardinalDirections.map(({ label, angle }) => {
          const dirRadius = radius - 15;
          const dirX = dirRadius * Math.cos((angle - 90) * (Math.PI / 180));
          const dirY = dirRadius * Math.sin((angle - 90) * (Math.PI / 180));
          
          return (
            <div 
              key={label}
              className="absolute text-xs font-medium text-muted-foreground"
              style={{
                left: size / 2 + dirX,
                top: size / 2 + dirY,
                transform: 'translate(-50%, -50%)'
              }}
            >
              {label}
            </div>
          );
        })}
        
        {/* Value indicator */}
        <div 
          className="absolute bg-primary text-primary-foreground rounded-full flex items-center justify-center z-10 shadow-md"
          style={{
            width: 24,
            height: 24,
            left: size / 2 + handleX,
            top: size / 2 + handleY,
            transform: 'translate(-50%, -50%)'
          }}
        />
        
        {/* Center */}
        <div 
          className="absolute bg-muted-foreground/20 rounded-full"
          style={{
            width: 6,
            height: 6,
            left: size / 2,
            top: size / 2,
            transform: 'translate(-50%, -50%)'
          }}
        />
        
        {/* Current value line */}
        <div 
          className="absolute bg-primary/30 h-0.5"
          style={{
            width: radius - 10,
            left: size / 2,
            top: size / 2,
            transform: `translate(0, -50%) rotate(${angle}deg)`,
            transformOrigin: 'left center'
          }}
        />
      </div>
      <div className="text-sm">
        <span className="font-medium">{value}°</span>
      </div>
    </div>
  );
};

export default CircularSlider;
