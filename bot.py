# ------ Core Setup ------
import logging
from pathlib import Path
from datetime import datetime

# ------ Data & Math ------
import numpy as np
import sympy as sp

# ------ Visualization ------
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from rich import print as rprint
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

# ------ ML/AI ------
from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

# ------ Utilities ------
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import qrcode
import pygame
import pyfiglet

# ------ Initialize GPT-Neo ------
def load_gpt_neo():
    """Load GPT-Neo model for local use"""
    model_name = "EleutherAI/gpt-neo-125M"  # Smaller model for local use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# ------ Helper Functions ------
def explain_code_snippet(code: str):
    """Interactive code explanation with syntax highlighting"""
    highlighted_code = highlight(code, PythonLexer(), TerminalFormatter())
    return f"```\n{highlighted_code}\n```"

def plot_equation(equation: str):
    """Interactive mathematical plotting"""
    x = sp.symbols('x')
    expr = sp.sympify(equation)
    lambdified = sp.lambdify(x, expr, 'numpy')
    
    x_vals = np.linspace(-10, 10, 400)
    y_vals = lambdified(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals)
    plt.title(f"Plot of {equation}")
    plt.grid(True)
    plt.savefig("equation_plot.png")
    plt.close()

def code_review_ai(code: str, model, tokenizer):
    """Automated code review using GPT-Neo"""
    prompt = f"Review this Python code:\n{code}\n\nFeedback:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=200, temperature=0.7)
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return feedback

def generate_qr_code(data: str):
    """Generate a QR code for learning resources"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save("learning_qr.png")

# ------ Telegram Bot Commands ------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    welcome_text = """
    🚀 Welcome to Code Wizard Bot! 🚀

    Available Commands:
    /start - Start the bot
    /help - Show help message
    /explain <code> - Explain Python code
    /plot <equation> - Plot a mathematical equation
    /review <code> - Get AI feedback on your code
    /qrcode <text> - Generate a QR code
    /tips - Show coding tips
    """
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a help message when the command /help is issued."""
    help_text = """
    🤖 How to use Code Wizard Bot:

    1. /explain <code> - Get syntax-highlighted explanation of your Python code.
    2. /plot <equation> - Plot a mathematical equation (e.g., x**2 + 3*x + 2).
    3. /review <code> - Get AI-powered feedback on your Python code.
    4. /qrcode <text> - Generate a QR code for any text or URL.
    5. /tips - Get useful coding tips.
    """
    await update.message.reply_text(help_text)

async def explain_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Explain Python code with syntax highlighting."""
    code = " ".join(context.args)
    if not code:
        await update.message.reply_text("Please provide some code to explain!")
        return
    explained_code = explain_code_snippet(code)
    await update.message.reply_text(explained_code, parse_mode="Markdown")

async def plot_equation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Plot a mathematical equation."""
    equation = " ".join(context.args)
    if not equation:
        await update.message.reply_text("Please provide an equation to plot!")
        return
    plot_equation(equation)
    await update.message.reply_photo(photo=open("equation_plot.png", "rb"))

async def code_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get AI feedback on Python code."""
    code = " ".join(context.args)
    if not code:
        await update.message.reply_text("Please provide some code to review!")
        return
    model, tokenizer = load_gpt_neo()
    feedback = code_review_ai(code, model, tokenizer)
    await update.message.reply_text(feedback)

async def generate_qrcode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a QR code."""
    data = " ".join(context.args)
    if not data:
        await update.message.reply_text("Please provide text or a URL!")
        return
    generate_qr_code(data)
    await update.message.reply_photo(photo=open("learning_qr.png", "rb"))

async def coding_tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show coding tips."""
    tips = """
    💡 Coding Tips:
    1. Use meaningful variable names.
    2. Write modular and reusable code.
    3. Always test your code with edge cases.
    4. Use version control (e.g., Git).
    5. Comment your code for clarity.
    6. Learn debugging techniques.
    7. Practice writing clean and efficient code.
    """
    await update.message.reply_text(tips)

# ------ Main Program ------
def main():
    """Start the bot."""
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    # Initialize bot with your token
    application = Application.builder().token("8005088864:AAEFpsT7vvXJUVL-Qcx2ocyGc9jSpF2gavM").build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("explain", explain_code))
    application.add_handler(CommandHandler("plot", plot_equation_command))
    application.add_handler(CommandHandler("review", code_review))
    application.add_handler(CommandHandler("qrcode", generate_qrcode))
    application.add_handler(CommandHandler("tips", coding_tips))

    # Start the bot
    application.run_polling()

# ------ Run Program ------
if __name__ == "__main__":
    main()
