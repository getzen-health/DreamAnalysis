import { apiRequest, resolveUrl } from "./queryClient";

export interface MoodAnalysisResult {
  mood: string;
  stressLevel: number;
  emotions: string[];
  recommendations: string[];
}

export interface DreamAnalysisResult {
  symbols: string[];
  emotions: Array<{
    emotion: string;
    intensity: number;
  }>;
  analysis: string;
}

export interface ChatResponse {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
}

/** Shape of a raw emotion entry returned by the dream analysis API (before normalization). */
interface RawEmotionEntry {
  emotion?: string;
  intensity?: number;
}

/** Shape of a chat record as returned by the API (timestamp is a serialized string/number). */
interface RawChatRecord {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: string | number;
}

export class OpenAIService {
  /**
   * Analyze mood and emotional state from text input
   */
  static async analyzeMood(text: string, userId: string): Promise<MoodAnalysisResult> {
    try {
      const response = await apiRequest("POST", "/api/analyze-mood", {
        text,
        userId
      });
      
      const result = await response.json();
      
      // Validate the response structure
      if (!result.mood || typeof result.stressLevel !== 'number') {
        throw new Error('Invalid response format from mood analysis');
      }
      
      return {
        mood: result.mood,
        stressLevel: Math.max(0, Math.min(100, result.stressLevel)),
        emotions: Array.isArray(result.emotions) ? result.emotions : [],
        recommendations: Array.isArray(result.recommendations) ? result.recommendations : []
      };
    } catch (error) {
      console.error('Mood analysis failed:', error);
      throw new Error(`Failed to analyze mood: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Analyze dream content and extract symbols, emotions, and interpretations
   */
  static async analyzeDream(dreamText: string, userId: string): Promise<DreamAnalysisResult> {
    try {
      const response = await apiRequest("POST", "/api/dream-analysis", {
        dreamText,
        userId
      });
      
      const result = await response.json();
      
      // Validate the response structure
      if (!result.symbols || !result.emotions || !result.aiAnalysis) {
        throw new Error('Invalid response format from dream analysis');
      }
      
      return {
        symbols: Array.isArray(result.symbols) ? result.symbols : [],
        emotions: Array.isArray(result.emotions) ? result.emotions.map((e: RawEmotionEntry) => ({
          emotion: e.emotion || 'Unknown',
          intensity: Math.max(0, Math.min(10, e.intensity || 0))
        })) : [],
        analysis: result.aiAnalysis
      };
    } catch (error) {
      console.error('Dream analysis failed:', error);
      throw new Error(`Failed to analyze dream: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Send a chat message to the AI companion
   */
  static async sendChatMessage(message: string, userId: string, history?: ChatResponse[]): Promise<ChatResponse> {
    try {
      const response = await apiRequest("POST", "/api/ai-chat", {
        message,
        userId,
        history: history?.map((h) => ({ message: h.message, isUser: h.isUser })) ?? [],
      });
      
      const result = await response.json();
      
      // Validate the response structure
      if (!result.id || !result.message || typeof result.isUser !== 'boolean') {
        throw new Error('Invalid response format from chat API');
      }
      
      return {
        id: result.id,
        message: result.message,
        isUser: result.isUser,
        timestamp: new Date(result.timestamp)
      };
    } catch (error) {
      console.error('Chat message failed:', error);
      throw new Error(`Failed to send chat message: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get chat history for a user
   */
  static async getChatHistory(userId: string): Promise<ChatResponse[]> {
    try {
      const response = await fetch(resolveUrl(`/api/ai-chat/${userId}`));
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (!Array.isArray(result)) {
        throw new Error('Invalid response format: expected array');
      }
      
      return result.map((chat: RawChatRecord) => ({
        id: chat.id,
        message: chat.message,
        isUser: chat.isUser,
        timestamp: new Date(chat.timestamp)
      }));
    } catch (error) {
      console.error('Failed to fetch chat history:', error);
      throw new Error(`Failed to fetch chat history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Export user data as CSV
   */
  static async exportUserData(userId: string): Promise<Blob> {
    try {
      const response = await fetch(resolveUrl(`/api/export/${userId}`));
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const blob = await response.blob();
      
      if (blob.size === 0) {
        throw new Error('Export returned empty file');
      }
      
      return blob;
    } catch (error) {
      console.error('Data export failed:', error);
      throw new Error(`Failed to export data: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Validate API key and connection
   */
  static async validateConnection(): Promise<boolean> {
    try {
      // Try a simple mood analysis with minimal text to test the connection
      const testResult = await this.analyzeMood("Test connection", "test-user");
      return !!testResult;
    } catch (error) {
      console.error('OpenAI connection validation failed:', error);
      return false;
    }
  }

  /**
   * Get available AI models (for future expansion)
   */
  static getAvailableModels(): string[] {
    // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
    return ["gpt-5", "gpt-4o", "gpt-4o-mini"];
  }

  /**
   * Check if OpenAI service is properly configured
   */
  static isConfigured(): boolean {
    // In a real implementation, this would check if API keys are available
    // For now, we assume the backend handles this validation
    return true;
  }
}

// Export convenience methods for common operations
export const analyzeMood = OpenAIService.analyzeMood;
export const analyzeDream = OpenAIService.analyzeDream;
export const sendChatMessage = OpenAIService.sendChatMessage;
export const getChatHistory = OpenAIService.getChatHistory;
export const exportUserData = OpenAIService.exportUserData;

// Default export
export default OpenAIService;
