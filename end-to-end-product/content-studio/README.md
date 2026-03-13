# Content Studio

## AI Multi-Agent System for Short-Form Video Production

> Generate **production-ready short-form video packages** using an AI
> multi-agent pipeline.\
> The system transforms a simple topic into a **script, storyboard,
> publishing assets, and video generation prompt** ready for tools like
> Runway, Pika, or Sora.

**Live Demo**
TBD

**Demo Video**
TBD

------------------------------------------------------------------------

# Overview

**Content Studio** is an AI-powered production pipeline that automates
the early stages of short-form video creation.

Instead of manually writing scripts, planning visuals, and preparing
social media assets, the system generates a **complete content package**
from a single topic input.

Example output includes:

-   Hook-optimized script
-   Structured storyboard
-   Thumbnail text and titles
-   Publishing assets
-   Video generation prompt
-   Editor brief

The system is designed as a **multi-agent workflow** orchestrated with
LangGraph and deployed as a cloud service.

------------------------------------------------------------------------

# System Architecture

User Input
(topic, audience, tone, platform)
        ↓
FastAPI API Layer
        ↓
LangGraph Multi-Agent Workflow
Script Writer Agents
        ↓
QA Selection Agent
        ↓
Revision Agent
        ↓
Storyboard Agent
        ↓
Thumbnail / Title Agent
        ↓
Music Agent
        ↓
Packaging Agent

↓

Production Ready Content Package

------------------------------------------------------------------------

# Multi-Agent Pipeline

The generation workflow is composed of specialized agents.

### Script Writer Agents

Generate multiple script candidates.

Responsibilities:

-   Hook generation
-   Narrative structure
-   Platform optimization

### QA Selection Agent

Evaluates candidate scripts and selects the strongest version based on:

-   Hook strength
-   Narrative clarity
-   Engagement potential

### Revision Agent

Improves the selected script by refining:

-   pacing
-   clarity
-   call-to-action

### Storyboard Agent

Transforms the script into a structured visual plan.

Example output:

Scene 1 --- 0-3s\
Visual: strong opening visual\
Voiceover: hook\
On-screen text: hook subtitle

### Title / Thumbnail Agent

Creates platform-optimized publishing assets.

Outputs:

-   multiple title options
-   thumbnail text
-   hook phrases

### Music Agent

Suggests background music style aligned with:

-   pacing
-   tone
-   emotional arc

### Packaging Agent

Produces the final structured output:

-   editor brief
-   video generation prompt
-   publishing assets

------------------------------------------------------------------------

# API Design

POST /api/content/generate

Example request:

``` json
{
  "topic": "The 3-Minute Rule for beating procrastination",
  "platform": "youtube_shorts",
  "audience": "Students and busy professionals",
  "tone": "Energetic and motivating",
  "duration_sec": 30,
  "reference_text": "Stop scrolling. If a task takes less than 3 minutes, do it now.",
  "language": "en"
}
```

------------------------------------------------------------------------

# Web Interface

A lightweight **Content Studio UI** is included.

/studio

Features:

-   Topic input
-   Platform toggle
-   Language toggle
-   Duration presets
-   Real-time generation

The UI displays:

-   final topic
-   script
-   storyboard
-   titles
-   thumbnail text
-   video generation prompt

------------------------------------------------------------------------

# Deployment

The system is deployed on **Google Cloud Run** using containerized
FastAPI services.

Infrastructure stack:

Docker
Google Artifact Registry
Google Cloud Run

Deployment URL:

https://content-studio-v1-910630079560.asia-northeast3.run.app/

------------------------------------------------------------------------

# Tech Stack

### AI / LLM

-   Anthropic Claude
-   LangGraph
-   LangChain Core

### Backend

-   FastAPI
-   Pydantic
-   Python 3.11

### Frontend

-   Jinja2 templates
-   Vanilla JavaScript

### Infrastructure

-   Docker
-   Google Cloud Run
-   Artifact Registry

------------------------------------------------------------------------

# Local Development

Clone the repository:

    git clone https://github.com/snhzyn/ai-portfolio.git
    cd content-studio

Install dependencies:

    poetry install

Run locally:

    uvicorn main:app --reload

Open:

    http://localhost:8000/studio

------------------------------------------------------------------------

# Future Improvements

Planned enhancements include:

-   RAG integration for topic grounding
-   analytics-driven script optimization
-   multi-platform adaptation
-   auto-video generation integration
-   creator workflow dashboard

------------------------------------------------------------------------

# Project Purpose

This project demonstrates:

-   **LLM multi-agent orchestration**
-   **AI-powered content production pipelines**
-   **end-to-end AI product deployment**

The goal is to explore how generative AI systems can **augment creative
workflows at scale.**
