import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { getOpenAIClient } from '../_lib/openai';
import { success, error, methodNotAllowed, badRequest } from '../_lib/response';
import { aiChats, healthMetrics } from '../../shared/schema';
import { eq, desc } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'POST') {
    try {
      const { message, userId, history } = req.body;

      if (!message || !userId) {
        return badRequest(res, 'Missing message or userId');
      }

      // DB is optional — skip persistence if DATABASE_URL is not configured
      let db: ReturnType<typeof getDb> | null = null;
      try { db = getDb(); } catch { /* no-op — run without persistence */ }

      if (db) {
        await db.insert(aiChats).values({ userId, message, isUser: true });
      }

      // Health context (best-effort — skip if DB unavailable)
      let healthContext = "";
      if (db) {
        try {
          const recentMetrics = await db
            .select()
            .from(healthMetrics)
            .where(eq(healthMetrics.userId, userId))
            .orderBy(desc(healthMetrics.timestamp))
            .limit(5);
          if (recentMetrics.length > 0) {
            healthContext = `Recent health data: Heart rate ${recentMetrics[0].heartRate}, Stress level ${recentMetrics[0].stressLevel}, Sleep quality ${recentMetrics[0].sleepQuality}`;
          }
        } catch { /* non-fatal */ }
      }

      const openai = getOpenAIClient();

      // Build conversation history for multi-turn context (last 20 messages)
      const historyMessages: { role: 'user' | 'assistant'; content: string }[] = Array.isArray(history)
        ? (history as Array<{ message: string; isUser: boolean }>)
            .slice(-20)
            .map((h) => ({ role: h.isUser ? 'user' : 'assistant', content: h.message }))
        : [];

      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: `You are an AI wellness companion for a Brain-Computer Interface system. You help users with mood analysis, stress relief, meditation, focus, and wellness. ${healthContext} Be warm, supportive, and give actionable advice. Keep responses clear and concise.`
          },
          ...historyMessages,
          { role: "user", content: message }
        ]
      });

      const aiResponse = response.choices[0].message.content || "I'm here to help you with your wellness journey.";

      // Persist AI reply if DB is available; otherwise return a synthetic record
      if (db) {
        const [newChat] = await db
          .insert(aiChats)
          .values({ userId, message: aiResponse, isUser: false })
          .returning();
        return success(res, newChat, 201);
      }

      // No DB — return a synthetic response object the client can use
      return success(res, {
        id: `ai-${Date.now()}`,
        userId,
        message: aiResponse,
        isUser: false,
        timestamp: new Date().toISOString(),
      }, 201);

    } catch (err) {
      console.error('Error processing chat message:', err);
      return error(res, 'Failed to process chat message');
    }
  }

  return methodNotAllowed(res, ['POST']);
}
