import asyncio
import os
import sys
from getpass import getpass

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, Provider

load_dotenv()


def verificar_env_keys():
    if not os.environ.get("OPENAI_API_KEY"):
        key = getpass("🔑 OPENAI_API_KEY: ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

    if not os.environ.get("ANTHROPIC_API_KEY"):
        key = getpass("🔑 ANTHROPIC_API_KEY: ").strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key

    if not os.environ.get("GOOGLE_API_KEY"):
        key = getpass("🔑 GOOGLE_API_KEY (aistudio.google.com/apikey): ").strip()
        if key:
            os.environ["GOOGLE_API_KEY"] = key


async def main():
    verificar_env_keys()

    # 1. Verificación previa de validación con Pydantic
    print("--- Prueba 1: Validación Pydantic (Temperatura Inválida) ---")
    try:
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5.0)
    except ValidationError as e:
        print("❌ Capturado correctamente ANTES de llamar a la API:\n", e)

    pregunta = [
        ChatMessage(
            role="user",
            content="¿Qué es la entropía? Respondé en 2 líneas.",
        )
    ]

    # 2. Prueba OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        print("\n--------------------------------------------------")
        print("🟢 PRUEBA OPENAI")
        print("--------------------------------------------------")
        config_openai = LLMConfig(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            openai_api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            temperature=0.7,
            max_tokens=200,
        )
        manager_openai = AsyncLLMManager(config_openai)

        res_openai = await manager_openai.generate(pregunta)
        print(
            "Respuesta Normal:",
            res_openai.content if not res_openai.error else f"❌ {res_openai.error}",
        )

        print("\nStreaming:")
        async for chunk in manager_openai.generate_stream(pregunta):
            print(chunk, end="", flush=True)
        print()

    # 3. Prueba Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n--------------------------------------------------")
        print("🟣 PRUEBA ANTHROPIC")
        print("--------------------------------------------------")
        config_anthropic = LLMConfig(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-5-5",
            anthropic_api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
            temperature=0.7,
            max_tokens=200,
        )
        manager_anthropic = AsyncLLMManager(config_anthropic)

        res_anthropic = await manager_anthropic.generate(pregunta)
        print(
            "Respuesta Normal:",
            res_anthropic.content
            if not res_anthropic.error
            else f"❌ {res_anthropic.error}",
        )

        print("\nStreaming:")
        async for chunk in manager_anthropic.generate_stream(pregunta):
            print(chunk, end="", flush=True)
        print()

    # 4. Prueba Gemini
    if os.environ.get("GOOGLE_API_KEY"):
        print("\n--------------------------------------------------")
        print("🔵 PRUEBA GEMINI")
        print("--------------------------------------------------")
        config_gemini = LLMConfig(
            provider=Provider.GEMINI,
            model="gemini-3.8-flash",  # Modelo estable y disponible
            google_api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
            temperature=1.0,
            max_tokens=200,
        )
        manager_gemini = AsyncLLMManager(config_gemini)

        res_gemini = await manager_gemini.generate(pregunta)
        print(
            "Respuesta Normal:",
            res_gemini.content if not res_gemini.error else f"❌ {res_gemini.error}",
        )

        print("\nStreaming:")
        async for chunk in manager_gemini.generate_stream(pregunta):
            print(chunk, end="", flush=True)
        print()

    # 5. Prueba de resiliencia (API Key Inválida)
    print("\n--------------------------------------------------")
    print("⚠️ PRUEBA DE CONTROL DE ERRORES (API Key Rota)")
    print("--------------------------------------------------")
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    manager_roto = AsyncLLMManager(config_rota)
    res_rota = await manager_roto.generate(pregunta)

    print("¿El programa siguió vivo sin crash?: ✅ Sí")
    print("Error capturado:", res_rota.error)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())