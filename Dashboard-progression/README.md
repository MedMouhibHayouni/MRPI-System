# Dashboard-Progression

Angular front-end for visualizing student learning data (P2 of the AiKup Tech internship). Converted from a React/Tailwind prototype into a standalone-components Angular app.

## Stack

- Angular 21 (standalone components, signals, `inject()`, `toSignal()`)
- Tailwind CSS v4 (via `@tailwindcss/postcss`)
- `lucide-angular` for icons
- `json-server` as the mock API (`db.json`)

## Project structure

```
src/app/
├── components/          # dashboard-layout, student-header, student-badges, subject-mastery-panel,
│                         # recommendation-list/-card, weekly-mission-list/-item, xp-progress-bar
├── models/               # student, recommendation, progress, student-profile interfaces
├── services/
│   ├── student.ts                  # StudentService — mock (json-server), TTL cache
│   ├── progress.ts                 # ProgressService — mock (json-server), TTL cache
│   ├── student-academic-profile.ts # StudentAcademicProfileService — static asset (public/students.json), cached once
│   ├── recommendation.ts           # RecommendationService — REAL FastAPI backend, cache keyed on request body
│   └── api-error-handler.ts        # ApiErrorHandler — maps HttpErrorResponse to user-facing messages
├── app.config.ts         # providers: router, HttpClient, zone change detection, lucide icons
└── app.routes.ts         # single dashboard route
src/environments/
├── environment.ts              # used by default build
└── environment.development.ts  # swapped in for the `development` build config
public/
└── students.json         # real learner dataset (from students.csv), served as a static asset
```

## Data sourcing

This app talks to **two different backends**, not one:

| Data | Source | Service | Why |
|---|---|---|---|
| Student identity (name/level/xp/badges) | `json-server` mock (`db.json`) | `StudentService` | No backend endpoint exists for this yet; stays constant regardless of selected student |
| Progress (mastery/missions) | `json-server` mock (`db.json`) | `ProgressService` | Same reason — mock only, constant |
| Learner profile (subject/weak_concept/academic_level/learning_style/past_interactions) | Static asset (`public/students.json`) | `StudentAcademicProfileService` | Real dataset; the backend has no GET-by-id endpoint that returns it |
| Recommendations | **Real FastAPI backend** (`POST /recommendations`) | `RecommendationService` | The actual CBF+CF hybrid engine — genuinely different output per learner profile |

Both API base URLs (`mockApiUrl` for json-server, `apiUrl` for the FastAPI backend) live in `src/environments/environment.ts` — no service hardcodes a host/port, so switching ports or environments means editing one file, not every service.

## Getting started

```bash
npm install

# Terminal 1 — mock API (student + progress) on http://localhost:3000
json-server --watch db.json --port 3000

# Terminal 2 — FastAPI recommendation backend on http://localhost:8000
# (see the MRPI-System repo — not part of this project)

# Terminal 3 — Angular dev server on http://localhost:4200
ng serve
```

If `json-server` ever starts on a different port (e.g. 3000 is already taken and it falls back to 3001), update `mockApiUrl` in `src/environments/environment.ts` — every mock-backed service picks it up automatically.

## Notes

- `StudentService` and `ProgressService` use a 5-minute TTL in-memory cache; `StudentAcademicProfileService` caches its one-time asset load via `shareReplay(1)`; `RecommendationService` caches per serialized request body, since two different learner profiles must never share a cached result.
- `ApiErrorHandler` centralizes all HTTP error → user-facing message mapping, so every service reports failures consistently.
- `DashboardLayout` combines all four sources into resource signals via `toSignal()`, so a failed request doesn't leave the whole page stuck — it renders the error state instead.
- Layout is responsive: tabs on mobile, two columns on tablet, three columns on desktop.