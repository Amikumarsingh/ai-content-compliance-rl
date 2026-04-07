# AI Compliance Dashboard

A modern, beautiful UI dashboard for the Content Compliance RL Environment.

![Dashboard Preview](./preview.png)

## Features

- **Dark Mode Design** - Premium glassmorphism UI with smooth animations
- **Real-time Evaluation** - Evaluate content for compliance violations
- **Task Simulation** - Run Easy, Medium, and Hard RL tasks with live terminal logs
- **History Tracking** - View and search past evaluations
- **Export Results** - Download evaluation results as JSON
- **Responsive Layout** - Works on desktop and tablet
- **Toast Notifications** - Real-time feedback for all actions

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Zustand** - State management
- **Sonner** - Toast notifications
- **Lucide React** - Beautiful icons

## Getting Started

### Prerequisites

- Node.js 18+ installed
- FastAPI backend running on port 7860

### Installation

1. **Navigate to the UI directory**
   ```bash
   cd ui-dashboard
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment**
   ```bash
   # Copy the example env file
   cp .env.example .env.local

   # Edit if needed (default is http://localhost:7860)
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open in browser**
   ```
   http://localhost:3000
   ```

## Connecting to Backend

The dashboard connects to your FastAPI backend automatically. Make sure your backend is running:

```bash
# From the project root
python main.py serve
```

The default API URL is `http://localhost:7860`. To change it, edit `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:7860
```

## API Endpoints

The dashboard expects these endpoints from the backend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/evaluate` | POST | Evaluate content (optional, falls back to mock) |
| `/spec` | GET | Environment spec |

## Building for Production

```bash
# Build the application
npm run build

# Start production server
npm start
```

## Project Structure

```
ui-dashboard/
├── src/
│   ├── app/
│   │   ├── globals.css      # Global styles & CSS variables
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main dashboard page
│   ├── components/
│   │   ├── Sidebar.tsx      # Left navigation sidebar
│   │   ├── TopBar.tsx       # Top header bar
│   │   ├── ContentInput.tsx # Content input section
│   │   ├── ResultCards.tsx  # Score, violations, metadata cards
│   │   ├── TaskPanel.tsx    # RL task simulation panel
│   │   ├── HistoryPanel.tsx # Evaluation history
│   │   └── Toast.tsx        # Toast notifications
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   ├── store.ts         # Zustand state management
│   │   └── utils.ts         # Utility functions
│   └── types/
│       └── index.ts         # TypeScript types
├── .env.local               # Environment variables
├── next.config.js           # Next.js configuration
├── tailwind.config.ts       # Tailwind CSS configuration
├── postcss.config.js        # PostCSS configuration
└── package.json             # Dependencies
```

## Design Features

### Color Scheme
- **Primary**: Violet to Fuchsia gradient
- **Safe**: Emerald green
- **Warning**: Amber yellow
- **Danger**: Red

### Animations
- Smooth page transitions
- Card scale-in effects
- Progress bar animations
- Pulsing glow effects
- Terminal-style log animations

### Glassmorphism
- Backdrop blur effects
- Semi-transparent backgrounds
- Subtle border gradients

## Customization

### Changing Colors
Edit `tailwind.config.ts` to modify the color scheme:

```ts
theme: {
  extend: {
    colors: {
      primary: {
        DEFAULT: 'hsl(263.4, 70%, 50.4%)', // Change this
      },
    },
  },
}
```

### Adding New Violation Types
Edit `src/lib/utils.ts`:

```ts
export function getViolationIconColor(type: string): string {
  const colors: Record<string, string> = {
    // Add new types here
    new_violation: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  }
  // ...
}
```

## Troubleshooting

### API Not Connecting
- Ensure the backend is running: `python main.py serve`
- Check the API URL in `.env.local`
- Verify CORS is enabled on the backend

### Build Errors
- Clear `.next` folder: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Styling Issues
- Ensure Tailwind CSS is properly configured
- Check that `globals.css` is imported in `layout.tsx`

## License

MIT

## Author

AI Compliance RL Environment Team
