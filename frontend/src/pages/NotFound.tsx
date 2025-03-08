
import React from 'react';
import { useLocation, useNavigate } from "react-router-dom";
import { Button } from '@/components/ui/button';
import { Home } from 'lucide-react';
import { useEffect } from "react";
import PageTransition from '@/components/animations/PageTransition';

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <PageTransition>
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="text-center max-w-md">
          <h1 className="text-7xl font-light mb-6">404</h1>
          <p className="text-xl text-muted-foreground mb-8">
            Oops! We couldn't find the page you're looking for.
          </p>
          <Button 
            size="lg" 
            className="rounded-full px-6"
            onClick={() => navigate('/')}
          >
            <Home className="mr-2 h-4 w-4" /> Back to Home
          </Button>
        </div>
      </div>
    </PageTransition>
  );
};

export default NotFound;
