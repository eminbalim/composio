/**
 * Philosophy Agent Example using Anthropic and Composio SDK
 *
 * This example demonstrates a philosophy agent that can engage in deep
 * philosophical discussions and use Composio tools to research topics.
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
import 'dotenv/config';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/**
 * Initialize Composio with the Anthropic provider
 */
const composio = new Composio({
  apiKey: process.env.COMPOSIO_API_KEY,
  provider: new AnthropicProvider({ cacheTools: true }),
});

const SYSTEM_PROMPT = `You are a philosophy agent with deep knowledge of philosophical traditions,
thinkers, and ideas spanning from ancient Greece to contemporary philosophy.

Your role is to:
- Engage thoughtfully with philosophical questions and ideas
- Draw connections between different philosophical traditions and thinkers
- Use available tools to research and fact-check philosophical claims
- Present balanced perspectives from multiple philosophical viewpoints
- Help users explore fundamental questions about existence, knowledge, ethics, and meaning

When discussing philosophy, cite relevant thinkers and their works, and explain
complex ideas in accessible language while maintaining intellectual rigor.`;

async function main() {
  try {
    console.log('Initializing Philosophy Agent...');

    /**
     * Get tools from Composio for philosophical research
     * (e.g., web search, Wikipedia lookup)
     */
    const tools = await composio.tools.get('default', {
      toolkits: ['TAVILY'],
    });
    console.log(`Fetched ${tools.length} research tools`);

    const userQuestion =
      'What is the relationship between Plato\'s Theory of Forms and modern mathematical Platonism? How do contemporary philosophers view this connection?';

    console.log('\nPhilosophy Agent');
    console.log('='.repeat(50));
    console.log(`Question: ${userQuestion}`);
    console.log('='.repeat(50));

    const messages: Anthropic.Messages.MessageParam[] = [
      {
        role: 'user',
        content: userQuestion,
      },
    ];

    /**
     * Agentic loop — continue until the model stops calling tools
     */
    let continueLoop = true;

    while (continueLoop) {
      const response = await anthropic.messages.create({
        model: 'claude-opus-4-5',
        max_tokens: 4096,
        system: SYSTEM_PROMPT,
        tools: tools as Anthropic.Messages.Tool[],
        messages,
      });

      messages.push({
        role: 'assistant',
        content: response.content,
      });

      const toolUseBlocks = response.content.filter(
        (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use'
      );

      if (toolUseBlocks.length > 0) {
        console.log(`\nUsing ${toolUseBlocks.length} research tool(s)...`);

        const toolResults = await composio.provider.handleToolCalls('default', response);
        messages.push(...toolResults);
      }

      if (response.stop_reason === 'end_turn' || toolUseBlocks.length === 0) {
        continueLoop = false;

        const finalText = response.content
          .filter((block): block is Anthropic.Messages.TextBlock => block.type === 'text')
          .map(block => block.text)
          .join('\n');

        console.log('\nAgent Response:');
        console.log('-'.repeat(50));
        console.log(finalText);
      }
    }
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
