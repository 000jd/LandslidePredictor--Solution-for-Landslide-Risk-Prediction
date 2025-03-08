
import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, AlertCircle, Info, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { toast } from "@/hooks/use-toast";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface RiskDisplayProps {
  riskLevel: 'low' | 'medium' | 'high';
  riskProbability: number;
  confidenceInterval: [number, number];
}

const RiskDisplay: React.FC<RiskDisplayProps> = ({
  riskLevel,
  riskProbability,
  confidenceInterval
}) => {
  const getRiskColor = () => {
    switch (riskLevel) {
      case 'low':
        return 'bg-risk-low text-white';
      case 'medium':
        return 'bg-risk-medium text-white';
      case 'high':
        return 'bg-risk-high text-white';
      default:
        return 'bg-muted text-foreground';
    }
  };

  const getRiskIcon = () => {
    switch (riskLevel) {
      case 'low':
        return <CheckCircle className="h-12 w-12" />;
      case 'medium':
        return <AlertCircle className="h-12 w-12" />;
      case 'high':
        return <AlertTriangle className="h-12 w-12" />;
      default:
        return <Info className="h-12 w-12" />;
    }
  };

  const getRiskText = () => {
    switch (riskLevel) {
      case 'low':
        return {
          title: 'Low Risk',
          description: 'The probability of a landslide occurring is relatively low based on the current conditions.',
          recommendations: [
            'Monitor conditions during extreme weather events',
            'Follow standard land management practices',
            'No immediate action required'
          ]
        };
      case 'medium':
        return {
          title: 'Medium Risk',
          description: 'There is a moderate probability of landslide occurrence under certain conditions.',
          recommendations: [
            'Increased monitoring during rainfall events',
            'Consider implementing preventive measures',
            'Have emergency plans prepared',
            'Avoid substantial modifications to slopes'
          ]
        };
      case 'high':
        return {
          title: 'High Risk',
          description: 'Current conditions indicate a high probability of landslide occurrence.',
          recommendations: [
            'Implement immediate risk mitigation measures',
            'Consider evacuation during heavy rainfall',
            'Engage geological experts for detailed assessment',
            'Avoid all activities that could destabilize slopes',
            'Install monitoring instruments if possible'
          ]
        };
      default:
        return {
          title: 'Unknown Risk',
          description: 'Risk level could not be determined.',
          recommendations: []
        };
    }
  };

  const downloadReport = () => {
    const riskInfo = getRiskText();
    
    // Create report content
    const reportContent = `
Landslide Risk Assessment Report
================================

RISK LEVEL: ${riskInfo.title.toUpperCase()}
Risk Probability: ${riskProbability.toFixed(1)}%
Confidence Interval: ${confidenceInterval[0].toFixed(1)}% - ${confidenceInterval[1].toFixed(1)}%

SUMMARY
-------
${riskInfo.description}

RECOMMENDED ACTIONS
------------------
${riskInfo.recommendations.map(rec => `• ${rec}`).join('\n')}

ASSESSMENT METHODOLOGY
---------------------
This risk assessment is based on machine learning models trained on historical landslide data, 
geographical parameters, and environmental conditions. The prediction uses a combination of 
input values to calculate the probability of landslide occurrence.

Report generated on: ${new Date().toLocaleString()}
    `;
    
    // Create blob from content
    const blob = new Blob([reportContent], { type: 'text/plain' });
    
    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `landslide-risk-report-${riskLevel}-${new Date().toISOString().split('T')[0]}.txt`;
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    // Show toast notification
    toast({
      title: "Report Downloaded",
      description: "Your risk assessment report has been downloaded successfully.",
      duration: 3000,
    });
  };

  const riskInfo = getRiskText();

  return (
    <div className="glass-card rounded-2xl overflow-hidden">
      <div className={`p-6 ${getRiskColor()}`}>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-light">{riskInfo.title}</h3>
            <p className="text-white/80 text-sm mt-1">Based on your input parameters</p>
          </div>
          <motion.div
            initial={{ rotate: -10, scale: 0.9 }}
            animate={{ rotate: 0, scale: 1 }}
            transition={{ duration: 0.5, type: "spring" }}
          >
            {getRiskIcon()}
          </motion.div>
        </div>
      </div>

      <div className="p-6 space-y-5">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm font-medium">Risk Probability</span>
            <span className="font-medium">{riskProbability.toFixed(1)}%</span>
          </div>
          <Progress value={riskProbability} className="h-2" />
          <div className="text-xs text-muted-foreground mt-1">
            Confidence Interval: {confidenceInterval[0].toFixed(1)}% - {confidenceInterval[1].toFixed(1)}%
          </div>
        </div>

        <div className="text-sm">{riskInfo.description}</div>

        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="recommendations">
            <AccordionTrigger className="text-primary">
              Recommended Actions
            </AccordionTrigger>
            <AccordionContent>
              <ul className="space-y-2 pl-5 list-disc">
                {riskInfo.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm">{rec}</li>
                ))}
              </ul>
            </AccordionContent>
          </AccordionItem>
          
          <AccordionItem value="methodology">
            <AccordionTrigger className="text-primary">
              Assessment Methodology
            </AccordionTrigger>
            <AccordionContent>
              <p className="text-sm text-muted-foreground">
                This risk assessment is based on machine learning models trained on historical landslide data, geographical parameters, and environmental conditions. The prediction uses a combination of your input values to calculate the probability of landslide occurrence.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <div className="mt-4">
          <Button className="w-full flex items-center justify-center" onClick={downloadReport}>
            <Download className="mr-2 h-4 w-4" /> Download Report
          </Button>
        </div>
      </div>
    </div>
  );
};

export default RiskDisplay;
