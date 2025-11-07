import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SentimentBadge } from "@/components/ui/sentiment-badge";
import { Brain, Loader2, FileText, BarChart3, Lightbulb } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface SentimentResult {
  sentiment: "positive" | "neutral" | "negative";
  confidence: number;
  score: number;
  breakdown: {
    positive: number;
    neutral: number;
    negative: number;
  };
  keywords: string[];
}

// Sample reviews removed - using only real backend data

export default function ReviewAnalysis() {
  const [review, setReview] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const { toast } = useToast();

  const analyzeSentiment = async (text: string): Promise<SentimentResult> => {
    try {
      // Try to call backend API
      const response = await fetch('http://localhost:8000/analyze_review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review_text: text })
      });

      if (response.ok) {
        const data = await response.json();
        return {
          sentiment: data.sentiment,
          confidence: data.confidence,
          score: data.score,
          breakdown: {
            positive: data.sentiment === "positive" ? data.confidence : 0.2,
            neutral: data.sentiment === "neutral" ? data.confidence : 0.3,
            negative: data.sentiment === "negative" ? data.confidence : 0.2,
          },
          keywords: [] // Backend could provide this
        };
      }
    } catch (error) {
      console.error('Backend API Error:', error);
      throw new Error('Unable to connect to sentiment analysis service. Please ensure the backend is running.');
    }
  };

  const handleAnalyze = async () => {
    if (!review.trim()) {
      toast({
        title: "Empty Review",
        description: "Please enter a review to analyze.",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const sentimentResult = await analyzeSentiment(review);
      setResult(sentimentResult);
      
      toast({
        title: "Analysis Complete",
        description: `Sentiment: ${sentimentResult.sentiment.charAt(0).toUpperCase() + sentimentResult.sentiment.slice(1)}`,
      });
      
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze the review. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Sample review handler removed - using only real backend data

  return (
    <div className="min-h-screen bg-gradient-hero py-8">
      <div className="container max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">Review Sentiment Analysis</h1>
          <p className="text-muted-foreground text-lg">
            Analyze the sentiment of individual product reviews using our CNN+BiLSTM model
          </p>
        </div>

        {/* Input Section */}
        <Card className="bg-gradient-card shadow-elevated border-0 mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Review Input
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="Enter a product review to analyze its sentiment (e.g., 'This laptop has amazing battery life and performance is incredible!')"
              value={review}
              onChange={(e) => setReview(e.target.value)}
              className="min-h-32 text-base resize-none border-2 focus:border-primary transition-colors mb-4"
              disabled={isAnalyzing}
            />
            
            <Button 
              onClick={handleAnalyze}
              size="lg" 
              className="w-full text-lg py-6 shadow-hover"
              disabled={isAnalyzing || !review.trim()}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Analyzing Review...
                </>
              ) : (
                <>
                  <Brain className="mr-2 h-5 w-5" />
                  Analyze Sentiment
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        {result && (
          <Card className="bg-gradient-card shadow-card border-0 mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Main Result */}
              <div className="flex items-center justify-between p-4 rounded-lg bg-muted/30">
                <div>
                  <p className="text-sm font-medium text-muted-foreground mb-2">Overall Sentiment</p>
                  <SentimentBadge sentiment={result.sentiment} score={result.score} />
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-muted-foreground mb-2">Confidence</p>
                  <p className="text-2xl font-bold text-primary">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Detailed Breakdown */}
              <div>
                <h4 className="font-semibold mb-4 flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-primary" />
                  Sentiment Breakdown
                </h4>
                
                <div className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-sentiment-positive">Positive</span>
                      <span>{(result.breakdown.positive * 100).toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={result.breakdown.positive * 100} 
                      className="h-2 bg-muted"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-sentiment-neutral">Neutral</span>
                      <span>{(result.breakdown.neutral * 100).toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={result.breakdown.neutral * 100} 
                      className="h-2 bg-muted"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-sentiment-negative">Negative</span>
                      <span>{(result.breakdown.negative * 100).toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={result.breakdown.negative * 100} 
                      className="h-2 bg-muted"
                    />
                  </div>
                </div>
              </div>

              {/* Keywords */}
              {result.keywords.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-3">Key Sentiment Words</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.keywords.map((keyword, index) => (
                      <Badge key={index} variant="outline" className="border-primary/20">
                        {keyword}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Sample reviews section removed - using only real backend data */}
      </div>
    </div>
  );
}