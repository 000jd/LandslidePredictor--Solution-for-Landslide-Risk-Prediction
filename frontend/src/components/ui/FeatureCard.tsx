
import React from 'react';
import { cn } from '@/lib/utils';

interface FeatureCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ 
  title, 
  children, 
  className,
  icon
}) => {
  return (
    <div className={cn(
      "glass-card rounded-xl p-4 glass-card-hover", 
      className
    )}>
      <div className="flex items-center space-x-2 mb-3">
        {icon && <div className="text-primary">{icon}</div>}
        <h3 className="text-base font-medium">{title}</h3>
      </div>
      <div>{children}</div>
    </div>
  );
};

export default FeatureCard;
