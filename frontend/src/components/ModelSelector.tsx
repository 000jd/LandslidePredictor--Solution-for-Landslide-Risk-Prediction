
import React from 'react';
import { Check, ChevronDown, BarChart3 } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import FeatureCard from './ui/FeatureCard';

interface ModelInfo {
  id: string;
  name: string;
  accuracy: number;
  trainingDate: string;
  type: string;
  description: string;
}

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onSelectModel
}) => {
  // Sample model data (in a real app, this would come from an API)
  const models: ModelInfo[] = [
    {
      id: "gradient-boost",
      name: "Gradient Boosting Model",
      accuracy: 92.5,
      trainingDate: "2023-10-15",
      type: "Gradient Boosting Classifier",
      description: "High-accuracy model trained on global landslide data with strong performance on diverse terrain types."
    },
    {
      id: "random-forest",
      name: "Random Forest Model",
      accuracy: 91.2,
      trainingDate: "2023-09-01",
      type: "Random Forest Classifier",
      description: "Balanced model with good generalization across different environmental conditions."
    },
    {
      id: "neural-network",
      name: "Deep Neural Network",
      accuracy: 93.8,
      trainingDate: "2023-11-20",
      type: "Deep Learning",
      description: "Advanced neural network trained on high-resolution terrain and climate data."
    }
  ];

  const selectedModelInfo = models.find(model => model.id === selectedModel) || models[0];

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  };

  return (
    <FeatureCard
      title="Select Prediction Model"
      icon={<BarChart3 className="h-5 w-5" />}
      className="w-full"
    >
      <div className="space-y-4">
        <Select value={selectedModel} onValueChange={onSelectModel}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {models.map(model => (
              <SelectItem key={model.id} value={model.id}>
                <div className="flex items-center justify-between w-full">
                  <span>{model.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {model.accuracy.toFixed(1)}% accuracy
                  </span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="bg-muted/40 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Model Type</span>
            <span className="text-sm">{selectedModelInfo.type}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Training Date</span>
            <span className="text-sm">{formatDate(selectedModelInfo.trainingDate)}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Accuracy</span>
            <span className="text-sm font-medium text-primary">{selectedModelInfo.accuracy}%</span>
          </div>
          
          <HoverCard>
            <HoverCardTrigger asChild>
              <button className="text-sm text-primary flex items-center mt-1 hover:underline">
                More info <ChevronDown className="h-3 w-3 ml-1" />
              </button>
            </HoverCardTrigger>
            <HoverCardContent className="w-80 text-sm">
              <div className="space-y-2">
                <h4 className="font-medium">{selectedModelInfo.name}</h4>
                <p className="text-sm text-muted-foreground">{selectedModelInfo.description}</p>
              </div>
            </HoverCardContent>
          </HoverCard>
        </div>
      </div>
    </FeatureCard>
  );
};

export default ModelSelector;
