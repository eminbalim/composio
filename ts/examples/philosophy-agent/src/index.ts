/**
 * Philosophy Agent - Interactive philosophical conversation agent
 *
 * A conversational agent powered by Claude via Composio that engages in
 * deep philosophical discussions. Covers topics like metaphysics, epistemology,
 * ethics, existentialism, and more.
 *
 * Features:
 * - Multi-turn interactive conversations in Turkish or English
 * - Streaming responses for real-time output
 * - Socratic dialogue method
 * - Web search tools for referencing philosophical works
 *
 * Required environment variables:
 * - COMPOSIO_API_KEY: Your Composio API key (get one at https://app.composio.dev)
 * - ANTHROPIC_API_KEY: Your Anthropic API key (get one at https://console.anthropic.com)
 *
 * Usage:
 *   bun src/index.ts
 */

import { Composio } from '@composio/core';
import { AnthropicProvider } from '@composio/anthropic';
import Anthropic from '@anthropic-ai/sdk';
import * as readline from 'readline';
import 'dotenv/config';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/**
 * Initialize Composio with the Anthropic provider
 */
const composio = new Composio({
  apiKey: process.env.COMPOSIO_API_KEY,
  provider: new AnthropicProvider(),
});

/**
 * System prompt that defines the philosophy agent's persona and expertise
 */
const PHILOSOPHY_SYSTEM_PROMPT = `Sen derin felsefi bilgiye sahip, düşündürücü sorular soran ve katılımcı diyalog yürüten bir felsefe ajanısın.

Uzmanlık alanların:
- Metafizik: varlık, gerçeklik, uzay, zaman ve nedensellik
- Epistemoloji: bilgi, inanç, doğrulama ve şüphecilik
- Etik: ahlaki değerler, normlar ve insan davranışı
- Siyaset felsefesi: adalet, güç, özgürlük ve toplum
- Estetik: güzellik, sanat ve estetik deneyim
- Varoluşçuluk: anlam, özgür irade, kimlik ve ölüm
- Mantık ve argümantasyon: tümdengelim, tümevarım, analoji

Felsefi gelenekler:
- Antik Yunan: Sokrates, Platon, Aristoteles, Epiktetos, Epikuros
- Aydınlanma: Descartes, Hume, Locke, Kant, Rousseau
- Modern: Nietzsche, Sartre, Camus, Heidegger, Wittgenstein
- Analitik felsefe: Russell, Frege, Quine, Rawls, Nozick
- Doğu felsefesi: Budizm, Taoizm, Konfüçyanizm, Hinduizm
- İslam felsefesi: İbn Sina, İbn Rüşd, Gazali, Farabi
- Türk düşünürler: Ziya Gökalp, Hilmi Ziya Ülken, Niyazi Berkes

Konuşma tarzın:
- Sokrates yöntemini uygula: sorularla düşünceleri derinleştir ve çelişkileri ortaya koy
- Karmaşık felsefi kavramları somut örneklerle açıkla
- Farklı felsefi geleneklerin bakış açılarını dengeli biçimde sun
- Kullanıcının kendi özgün düşüncelerini geliştirmesini teşvik et
- Hem Türkçe hem İngilizce felsefe terminolojisini yerinde kullan
- Güncel toplumsal sorunlara felsefi perspektiften yaklaş
- Her yanıtın sonunda bir düşündürücü soru veya alıntı paylaş

Önemli: Kullanıcı hangi dilde yazarsa o dilde yanıt ver.`;

/**
 * Print a styled welcome banner
 */
function printBanner(): void {
  console.log('\n╔══════════════════════════════════════════════════╗');
  console.log('║           FELSEFE AJANI  /  PHILOSOPHY AGENT        ║');
  console.log('║         Powered by Claude via Composio SDK          ║');
  console.log('╚══════════════════════════════════════════════════╝\n');
  console.log('Merhaba! Benimle felsefi konular hakkında konuşabilirsin.');
  console.log('Hello! You can discuss philosophical topics with me.\n');
  console.log('Konu önerileri / Topic suggestions:');
  console.log('  • "Özgür irade var mı?" / "Does free will exist?"');
  console.log('  • "Ahlakın temeli nedir?" / "What is the foundation of morality?"');
  console.log('  • "Gerçeklik nedir?" / "What is reality?"');
  console.log('  • "Yaşamın anlamı nedir?" / "What is the meaning of life?"');
  console.log('\nÇıkmak için: "çıkış", "exit" veya "quit"');
  console.log('─'.repeat(54));
}

/**
 * Main function that runs the interactive philosophy agent
 */
async function main(): Promise<void> {
  printBanner();

  /**
   * Try to fetch web search tools from Composio for referencing philosophical works.
   * Falls back gracefully if tools are unavailable.
   */
  let tools: Anthropic.Messages.Tool[] = [];
  try {
    const fetchedTools = await composio.tools.get('default', 'TAVILY_SEARCH');
    if (fetchedTools.length > 0) {
      tools = fetchedTools as Anthropic.Messages.Tool[];
      console.log(`\n[Composio: ${tools.length} araç yüklendi / tool(s) loaded]\n`);
    }
  } catch {
    // Tools not available, continue without them
  }

  const messages: Anthropic.Messages.MessageParam[] = [];

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuestion = (prompt: string): Promise<string> =>
    new Promise(resolve => rl.question(prompt, resolve));

  /**
   * Main conversation loop
   */
  while (true) {
    const userInput = await askQuestion('\nSen / You: ');
    const trimmed = userInput.trim();

    if (!trimmed) continue;

    if (['çıkış', 'cikis', 'exit', 'quit', 'q'].includes(trimmed.toLowerCase())) {
      console.log(
        '\n"Sorgulanmamış bir hayat yaşanmaya değer değildir." — Sokrates',
      );
      console.log('"The unexamined life is not worth living." — Socrates\n');
      rl.close();
      break;
    }

    messages.push({ role: 'user', content: trimmed });

    try {
      process.stdout.write('\nAjan / Agent: ');

      /**
       * Use streaming for real-time philosophical discourse
       */
      const requestParams: Anthropic.Messages.MessageStreamParams = {
        model: 'claude-3-7-sonnet-latest',
        max_tokens: 2048,
        system: PHILOSOPHY_SYSTEM_PROMPT,
        messages,
        ...(tools.length > 0 ? { tools } : {}),
      };

      let fullResponse = '';
      let toolCallsMade = false;

      const stream = anthropic.messages.stream(requestParams);

      for await (const event of stream) {
        if (
          event.type === 'content_block_delta' &&
          event.delta.type === 'text_delta'
        ) {
          process.stdout.write(event.delta.text);
          fullResponse += event.delta.text;
        }
      }

      const finalMessage = await stream.finalMessage();

      /**
       * Handle tool calls if any were made (e.g., web search for philosophical references)
       */
      if (finalMessage.stop_reason === 'tool_use') {
        toolCallsMade = true;
        messages.push({ role: 'assistant', content: finalMessage.content });

        const toolResults = await composio.provider.handleToolCalls('default', finalMessage);
        messages.push(...toolResults);

        /**
         * Get the final response after tool execution
         */
        process.stdout.write('\n');
        const followUp = await anthropic.messages.create({
          model: 'claude-3-7-sonnet-latest',
          max_tokens: 2048,
          system: PHILOSOPHY_SYSTEM_PROMPT,
          messages,
        });

        fullResponse = followUp.content
          .filter(block => block.type === 'text')
          .map(block => (block as Anthropic.Messages.TextBlock).text)
          .join('\n');

        console.log(fullResponse);
        messages.push({ role: 'assistant', content: fullResponse });
      } else {
        if (!toolCallsMade) {
          console.log();
          messages.push({ role: 'assistant', content: fullResponse });
        }
      }

      /**
       * Keep conversation history manageable (last 30 exchanges = 60 messages)
       */
      if (messages.length > 60) {
        messages.splice(0, 2);
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error(`\n[Hata / Error]: ${error.message}`);
      } else {
        console.error('\n[Hata / Error]: Beklenmeyen bir hata oluştu.');
      }
    }
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
