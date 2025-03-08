
import React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from 'lucide-react';

interface InfoTooltipProps {
  content: string | React.ReactNode;
  children?: React.ReactNode;
}

const InfoTooltip: React.FC<InfoTooltipProps> = ({ content, children }) => {
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <span className="inline-flex items-center cursor-help">
          {children || <Info className="h-4 w-4 text-muted-foreground ml-1" />}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-sm p-3 glass-card">
        {content}
      </TooltipContent>
    </Tooltip>
  );
};

export default InfoTooltip;
