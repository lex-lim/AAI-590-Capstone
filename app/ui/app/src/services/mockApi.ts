// Mock API functions for face authentication and intent classification

export interface IntentResult {
  intent: string;
  confidence: number;
}

export interface MCPServer {
  name: string;
  description: string;
}

/**
 * Mock face authentication - simulates authentication process
 */
export async function mockFaceAuth(): Promise<{ success: boolean }> {
  // Simulate processing delay
  await new Promise((resolve) => setTimeout(resolve, 3000));
  return { success: true };
}

/**
 * Available domains from domains.json
 */
const DOMAINS = [
  'banking',
  'credit_cards',
  'kitchen_and_dining',
  'home',
  'auto_and_commute',
  'travel',
  'utility',
  'work',
  'small_talk',
  'meta',
];

/**
 * Mock intent classification - returns a random domain from domains.json
 */
export async function mockClassifyIntent(_text: string): Promise<IntentResult> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Return a random domain from the 10 available domains
  const randomDomain = DOMAINS[Math.floor(Math.random() * DOMAINS.length)];
  const confidence = 0.85 + Math.random() * 0.1; // Random confidence between 0.85 and 0.95

  return { intent: randomDomain, confidence };
}

/**
 * Get activated MCP servers based on classified intent (domain)
 */
export function mockGetActivatedServers(intent: string): MCPServer[] {
  const serverMap: Record<string, MCPServer[]> = {
    banking: [
      { name: 'Banking API', description: 'Manages bank accounts, transfers, and transactions' },
      { name: 'Account Service', description: 'Handles account operations and balance queries' },
    ],
    credit_cards: [
      { name: 'Credit Card Service', description: 'Manages credit card accounts and rewards' },
      { name: 'Credit Score API', description: 'Provides credit score and limit information' },
    ],
    kitchen_and_dining: [
      { name: 'Recipe Service', description: 'Provides recipes and cooking instructions' },
      { name: 'Restaurant API', description: 'Handles restaurant reservations and reviews' },
    ],
    home: [
      { name: 'Smart Home Controller', description: 'Controls home automation and devices' },
      { name: 'Music Service', description: 'Manages music playback and playlists' },
      { name: 'Calendar Service', description: 'Handles calendar and reminder management' },
    ],
    auto_and_commute: [
      { name: 'Navigation Service', description: 'Provides directions and traffic information' },
      { name: 'Vehicle Maintenance API', description: 'Tracks vehicle maintenance and diagnostics' },
      { name: 'Ride Service', description: 'Manages ride-sharing and transportation' },
    ],
    travel: [
      { name: 'Travel Booking API', description: 'Handles flight and hotel reservations' },
      { name: 'Translation Service', description: 'Provides language translation services' },
      { name: 'Travel Info Service', description: 'Offers travel alerts and destination information' },
    ],
    utility: [
      { name: 'Weather Service', description: 'Provides weather data and forecasts' },
      { name: 'Time Service', description: 'Handles time, date, and timezone queries' },
      { name: 'Calculator Service', description: 'Performs calculations and conversions' },
    ],
    work: [
      { name: 'HR Service', description: 'Manages PTO, payroll, and employee information' },
      { name: 'Meeting Scheduler', description: 'Handles meeting scheduling and calendar management' },
      { name: 'Workplace API', description: 'Provides work-related utilities and information' },
    ],
    small_talk: [
      { name: 'Conversation Service', description: 'Handles general conversation and greetings' },
    ],
    meta: [
      { name: 'Settings Manager', description: 'Manages system settings and preferences' },
      { name: 'Device Sync Service', description: 'Handles device synchronization and configuration' },
    ],
  };

  return serverMap[intent] || [];
}

