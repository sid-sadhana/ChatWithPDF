# Frontend — ChatWithPDF UI

React + Tailwind CSS + Framer Motion frontend for ChatWithPDF.

## Layout

```
frontend/
├── src/
│   ├── components/
│   │   ├── App.js
│   │   ├── Sidebar.jsx
│   │   ├── ChatWindow.jsx
│   │   ├── MessageList.jsx
│   │   ├── MessageBubble.jsx
│   │   ├── ChatInput.jsx
│   │   ├── EmptyState.jsx
│   │   └── Toast.jsx
│   ├── hooks/
│   │   └── useAutoScroll.js
│   ├── services/
│   │   └── api.js
│   ├── index.js
│   └── index.css
├── package.json
├── tailwind.config.js
├── nginx.conf
└── Dockerfile
```

## Local development

```bash
cd frontend
npm install
npm start
```

Dev server runs on http://localhost:3000. API calls go to http://localhost:8000.

## Production build

```bash
npm run build
```

In Docker, the static build is served by nginx which also reverse-proxies `/api` to the backend.

## Configuration

| Variable | Description |
| --- | --- |
| `REACT_APP_API_BASE_URL` | Backend URL. Defaults to `http://localhost:8000`. |
