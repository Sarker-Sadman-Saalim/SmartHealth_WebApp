# obesity_chatbot.py
from __future__ import annotations

import re
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ObesityChatbot:
    """
    Production-style hybrid chatbot:
    - Rules/router for common intents (fast + grounded)
    - Phi-3 only for complex questions (more human-like)
    - Avoids context echo + HTML garbage
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        print(f"[Chatbot] Loading {model_name} on {self.device}...")

        # IMPORTANT: trust_remote_code needed for some Phi models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()
        print("[Chatbot] ✓ Ready")

        self.food_keywords = {
            "rice","chicken","beef","fish","egg","eggs","bread","pasta","noodles","pizza","burger","fries",
            "coke","soda","juice","tea","coffee","sweets","chocolate","cake","biscuit","chips",
            "salad","vegetable","veggies","beans","lentils","milk","yogurt","oats"
        }

    # ======================================================
    # Public
    # ======================================================
    def generate_response(self, user_message: str, context: Dict[str, Any]) -> str:
        msg = (user_message or "").strip()
        if not msg:
            return "Ask me about **diet**, **exercise**, **why you got this result**, or **what to do next**."

        # 1) Handle expectation follow-ups (short numeric replies)
        expect = context.get("expect")
        follow = self._handle_expected_reply(msg, expect, context)
        if follow:
            return follow

        # 2) Fast router (professional deterministic)
        intent = self._classify_intent(msg)
        quick = self._quick_reply(intent)
        if quick:
            return quick

        factual = self._factual_reply(intent, context)
        if factual:
            return factual

        routed = self._router_reply(intent, msg, context)
        if routed:
            return routed

        # 3) Phi-3 fallback (only when needed)
        prompt = self._build_prompt(msg, context)
        out = self._phi_generate(prompt)
        out = self._clean(out)

        # Guard: if model echoes context or outputs garbage, fallback
        if self._looks_like_echo(out) or len(out) < 40:
            return self._fallback(context)

        return out

    # ======================================================
    # Expected reply handler (solves "2" problem)
    # ======================================================
    def _handle_expected_reply(self, msg: str, expect: Optional[str], ctx: Dict[str, Any]) -> Optional[str]:
        m = msg.strip().lower()

        if expect == "EXPECT_MEALS_PER_DAY":
            if m in {"1", "once"}:
                return (
                    "Great — **once per day** is reasonable.\n\n"
                    "To improve faster:\n"
                    "• Keep rice to 1 fist-sized portion\n"
                    "• Add vegetables (half plate)\n"
                    "• Keep chicken grilled/boiled/air-fried\n\n"
                    "Want a **full daily diet plan** or an **exercise plan** next?"
                )
            if m in {"2", "twice"}:
                return (
                    "**Twice per day** can still work if portions are controlled.\n\n"
                    "Do this:\n"
                    "• Slightly reduce rice at one meal\n"
                    "• Add vegetables to both meals\n"
                    "• Avoid sugary sauces / deep frying\n\n"
                    "Tell me: is your goal **weight loss** or **healthy maintenance**?"
                )
            if m in {"3", "3+", "three"}:
                return (
                    "Eating it **3+ times/day** can slow progress.\n\n"
                    "Better approach:\n"
                    "• Keep one meal rice+chicken\n"
                    "• Other meals: vegetables + protein (less carbs)\n"
                    "• Add daily walking\n\n"
                    "Want a **7-day meal plan**?"
                )
        return None

    # ======================================================
    # Intent classification (strict)
    # ======================================================
    def _classify_intent(self, msg: str) -> str:
        m = msg.lower().strip()

        if m in {"hi", "hello", "hey", "hii", "hola"}:
            return "greeting"
        if m in {"thanks", "thank you", "thx"}:
            return "thanks"

        if re.search(r"\b(age)\b", m): return "age"
        if re.search(r"\b(height)\b", m): return "height"
        if re.search(r"\b(weight)\b", m): return "weight"
        if re.search(r"\b(bmi)\b", m): return "bmi"

        if re.search(r"\b(why|reason|explain)\b", m): return "why"
        if re.search(r"\b(diet|meal|food|eat|nutrition)\b", m): return "diet"
        if re.search(r"\b(exercise|workout|activity|gym)\b", m): return "exercise"
        if re.search(r"\b(what should i do now|now what|next step|next steps)\b", m): return "next_steps"

        tokens = set(re.findall(r"[a-z']+", m))
        if len(tokens & self.food_keywords) >= 1:
            return "diet_followup"

        return "other"

    # ======================================================
    # Quick replies
    # ======================================================
    def _quick_reply(self, intent: str) -> Optional[str]:
        if intent == "greeting":
            return (
                "Hi! 👋 I’m your health guidance assistant.\n\n"
                "Try:\n"
                "• What should I do now?\n"
                "• Give me a diet plan\n"
                "• Give me an exercise plan\n"
                "• Why did I get this result?"
            )
        if intent == "thanks":
            return "You’re welcome! Want a **diet plan**, **exercise plan**, or **both**?"
        return None

    # ======================================================
    # Factual replies (never hallucinate)
    # ======================================================
    def _factual_reply(self, intent: str, ctx: Dict[str, Any]) -> Optional[str]:
        if intent == "age" and ctx.get("age") is not None:
            return f"Your age is **{ctx.get('age')}** years."
        if intent == "height" and ctx.get("height") is not None:
            return f"Your height is **{ctx.get('height')} m**."
        if intent == "weight" and ctx.get("weight") is not None:
            return f"Your weight is **{ctx.get('weight')} kg**."
        if intent == "bmi" and ctx.get("bmi") is not None:
            try:
                return f"Your BMI is **{float(ctx.get('bmi')):.1f}**."
            except Exception:
                return f"Your BMI is **{ctx.get('bmi')}**."
        return None

    # ======================================================
    # Router replies (fast, professional)
    # ======================================================
    def _router_reply(self, intent: str, msg: str, ctx: Dict[str, Any]) -> Optional[str]:
        prediction = ctx.get("prediction", "Unknown")
        factors = (ctx.get("top_factors") or [])[:3]
        main = factors[0] if factors else "overall lifestyle balance"

        if intent == "why":
            bullets = "\n".join([f"• {f}" for f in factors]) if factors else "• Multiple lifestyle factors"
            return (
                f"**Your result:** {prediction}\n"
                f"**Top driver:** {main}\n\n"
                f"Key factors:\n{bullets}\n\n"
                f"Want a **diet plan** or **exercise plan** first?"
            )

        if intent == "next_steps":
            return (
                f"Here’s a practical plan for **{prediction}**:\n\n"
                f"1) Start with **{main}**\n"
                f"2) Walk 20–30 min daily (or 5 days/week)\n"
                f"3) Meals: vegetables + protein each meal\n"
                f"4) Reduce sugary drinks/fast food\n"
                f"5) Sleep 7–9h and track weekly\n\n"
                f"Tell me: do you want **diet**, **exercise**, or a **7-day plan**?"
            )

        if intent == "diet":
            # set expectation externally via app.py
            return (
                f"**Diet plan (simple + realistic) for {prediction}:**\n"
                f"• Half plate vegetables at lunch & dinner\n"
                f"• Protein every meal (eggs/fish/chicken/beans)\n"
                f"• 1 fist-sized carb portion per meal (rice/bread/pasta)\n"
                f"• Replace sugary drinks with water (~2L/day)\n"
                f"• Snacks: fruit, yogurt, nuts\n\n"
                f"Now tell me what you usually eat (example: rice and chicken)."
            )

        if intent == "diet_followup":
            return (
                "Nice — that’s a good base. Here’s how to make it healthier:\n\n"
                "• **Rice**: keep it to ~1 fist-sized portion per meal\n"
                "• **Chicken**: prefer grilled/boiled/air-fried (avoid deep fried)\n"
                "• **Add vegetables**: half your plate should be vegetables\n"
                "• **Oil/sauces**: reduce oil; avoid sugary sauces\n\n"
                "Quick question: how many times per day do you eat this?\n"
                "Reply with **1**, **2**, or **3+**."
            )

        if intent == "exercise":
            faf = ctx.get("faf", "N/A")
            return (
                "**Exercise plan (safe start):**\n"
                "• Walk 20–30 min, 5 days/week\n"
                "• Strength training 2 days/week (squats, wall pushups, bands)\n"
                "• Increase slowly week by week\n\n"
                f"Your current activity (FAF): **{faf}** days/week.\n"
                "Do you prefer **home workouts** or **gym**?"
            )

        return None

    # ======================================================
    # Phi-3 prompt + generation
    # ======================================================
    def _build_prompt(self, user_msg: str, ctx: Dict[str, Any]) -> str:
        # Keep context concise (avoid echo)
        top = ", ".join((ctx.get("top_factors") or [])[:3])
        bmi = ctx.get("bmi")
        bmi_str = f"{float(bmi):.1f}" if bmi is not None else "N/A"

        system = (
            "You are a professional health guidance assistant. "
            "Give safe, practical lifestyle advice. Not a medical diagnosis. "
            "Do NOT repeat the context. Be concise and actionable."
        )

        context_block = (
            f"Prediction: {ctx.get('prediction')}\n"
            f"BMI: {bmi_str}\n"
            f"Age: {ctx.get('age')}\n"
            f"FAF: {ctx.get('faf')}\n"
            f"FCVC: {ctx.get('fcvc')}\n"
            f"CH2O: {ctx.get('ch2o')}\n"
            f"Top factors: {top}\n"
        )

        # Phi-3 chat style
        prompt = (
            f"<|system|>\n{system}\n"
            f"<|user|>\nContext:\n{context_block}\nQuestion: {user_msg}\n"
            f"<|assistant|>\n"
        )
        return prompt

    def _phi_generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,         # deterministic for safety
                num_beams=1,             # faster
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Extract after assistant marker
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>", 1)[-1].strip()
        return text.strip()

    # ======================================================
    # Guards + fallback
    # ======================================================
    def _clean(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _looks_like_echo(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["prediction:", "bmi:", "age:", "faf:", "fcvc:", "ch2o:", "context:"])

    def _fallback(self, ctx: Dict[str, Any]) -> str:
        return (
            f"Based on your category (**{ctx.get('prediction','Unknown')}**), focus on:\n"
            "• Regular walking + strength training\n"
            "• More vegetables + protein\n"
            "• Smaller carb portions\n"
            "• Good sleep and hydration\n\n"
            "Ask me for a **diet plan** or **exercise plan**."
        )
