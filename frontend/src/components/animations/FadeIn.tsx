
import React from 'react';
import { motion } from 'framer-motion';

interface FadeInProps {
  children: React.ReactNode;
  delay?: number;
  direction?: 'up' | 'down' | 'left' | 'right' | 'none';
  className?: string;
}

const FadeIn: React.FC<FadeInProps> = ({ 
  children, 
  delay = 0, 
  direction = 'up',
  className = '' 
}) => {
  const getDirectionValues = () => {
    switch (direction) {
      case 'up':
        return { y: 20, x: 0 };
      case 'down':
        return { y: -20, x: 0 };
      case 'left':
        return { y: 0, x: 20 };
      case 'right':
        return { y: 0, x: -20 };
      default:
        return { y: 0, x: 0 };
    }
  };

  const { y, x } = getDirectionValues();

  return (
    <motion.div
      initial={{ opacity: 0, y, x }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      transition={{ 
        duration: 0.5, 
        delay: delay, 
        ease: [0.22, 1, 0.36, 1] 
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

export default FadeIn;
