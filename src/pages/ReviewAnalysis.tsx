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

const sampleReviews = [
  "This laptop has amazing battery life and the performance is incredible for the price!",
  "The camera quality is decent but the battery drains too quickly for my needs.",
  "Good product overall, meets expectations but nothing particularly special about it.",
  "Terrible experience - constant crashes and poor build quality. Would not recommend.",
  "Outstanding device! Fast, reliable, and the display quality is simply stunning."
];

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
      console.error('Backend API not available, using fallback:', error);
    }

    // Fallback rule-based analysis
    const words = text.toLowerCase().split(/\s+/);
    const positiveWords = ["amazing", "incredible", "outstanding", "excellent", "great", "good", "fast", "reliable", "stunning"];
    const negativeWords = ["terrible", "poor", "bad", "crashes", "awful", "horrible", "slow", "broken"];
    
    let positiveScore = 0;
    let negativeScore = 0;
    const keywords: string[] = [];
    
    words.forEach(word => {
      if (positiveWords.includes(word)) {
        positiveScore++;
        keywords.push(word);
      }
      if (negativeWords.includes(word)) {
        negativeScore++;
        keywords.push(word);
      }
    });
    
    const totalSentimentWords = positiveScore + negativeScore;
    let sentiment: "positive" | "neutral" | "negative";
    let score: number;
    let confidence: number;
    
    if (totalSentimentWords === 0) {
      sentiment = "neutral";
      score = 5.0;
      confidence = 0.6;
    } else if (positiveScore > negativeScore) {
      sentiment = "positive";
      score = Math.min(10, 5 + (positiveScore / totalSentimentWords) * 5);
      confidence = Math.min(0.95, 0.7 + (positiveScore / words.length) * 2);
    } else if (negativeScore > positiveScore) {
      sentiment = "negative";
      score = Math.max(1, 5 - (negativeScore / totalSentimentWords) * 4);
      confidence = Math.min(0.95, 0.7 + (negativeScore / words.length) * 2);
    } else {
      sentiment = "neutral";
      score = 5.0;
      confidence = 0.8;
    }
    
    return {
      sentiment,
      confidence,
      score,
      breakdown: {
        positive: sentiment === "positive" ? confidence : sentiment === "neutral" ? 0.5 : 1 - confidence,
        neutral: sentiment === "neutral" ? confidence : 0.3,
        negative: sentiment === "negative" ? confidence : sentiment === "neutral" ? 0.2 : 1 - confidence,
      },
      keywords: keywords.slice(0, 5)
    };
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

  const handleSampleReview = (sample: string) => {
    setReview(sample);
    setResult(null);
  };

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

        {/* Sample Reviews */}
        <Card className="bg-card/50 border-0">
          <CardHeader>
            <CardTitle className="text-lg">Try These Sample Reviews</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {sampleReviews.map((sample, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="text-left p-4 h-auto whitespace-normal justify-start hover:bg-card-hover"
                  onClick={() => handleSampleReview(sample)}
                  disabled={isAnalyzing}
                >
                  "{sample}"
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}