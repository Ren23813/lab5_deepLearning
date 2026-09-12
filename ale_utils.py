"""
Módulo de funciones reutilizables para crear entornos de Gymnasium/ALE,
ejecutar agentes (aleatorio o basado en reglas) y grabar video de las
partidas.
"""

import os
import numpy as np
import gymnasium as gym
import ale_py

# Registrar los entornos de ALE (ALE/SpaceInvaders-v5, etc.) en Gymnasium.
gym.register_envs(ale_py)


def crear_entorno(nombre_entorno, video_folder=None, name_prefix="video",
                   episode_trigger=None, render_mode="rgb_array", **kwargs):
    """
    Crea y retorna un entorno de Gymnasium.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno a crear.  "ALE/SpaceInvaders-v5".
    video_folder : str, opcional
        Carpeta donde se guardarán los videos. Si se especifica, el entorno
        se envuelve con gymnasium.wrappers.RecordVideo.
    name_prefix : str
        Prefijo para el nombre de los archivos de video generados.
    episode_trigger : callable, opcional
        Función que recibe el número de episodio (int) y retorna True/False
        indicando si ese episodio debe grabarse. Por defecto graba todos
        los episodios (lambda ep: True).
    render_mode : str
        Modo de renderizado del entorno. Debe ser "rgb_array" para poder
        grabar video con RecordVideo.
    **kwargs :
        Argumentos adicionales que se pasan directamente a gymnasium.make
        (p. ej. obs_type="ram", frameskip=4, repeat_action_probability=0.0,
        full_action_space=False).

    Retorna
    -------
    env : gymnasium.Env
        Entorno de Gymnasium, envuelto en RecordVideo si video_folder no es
        None.
    """
    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
        )

    return env


def agente_aleatorio(observation, env):
    """
    Agente base (baseline) que no requiere entrenamiento.

    Recibe la observación actual y el entorno, y retorna una acción
    muestreada uniformemente al azar del espacio de acciones del entorno.

    Parámetros
    ----------
    observation : np.ndarray
        Observación actual retornada por el entorno (no se utiliza, se
        incluye solo para cumplir con la firma común de "función de agente").
    env : gymnasium.Env
        Entorno del cual se toma el espacio de acciones.

    Retorna
    -------
    action : int
        Acción muestreada de env.action_space.
    """
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """
    Agente basado en una regla/política simple (sin entrenamiento) para
    ALE/SpaceInvaders-v5, usado como referencia adicional al azar.

    Regla: dispara la mayoría del tiempo y, mientras dispara, alterna el
    movimiento horizontal (izquierda/derecha) para barrer la pantalla,
    en lugar de tomar acciones solo aleatorias.

    Funciona con el espacio de acciones por defecto de Space Invaders:
    [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE].
    Si el entorno tiene menos de 6 acciones (otro juego), cae de vuelta a
    una acción aleatoria para no fallar.

    Parámetros
    ----------
    observation : np.ndarray
        Observación actual (no se analiza en detalle; la regla es simple
        e independiente del contenido visual).
    env : gymnasium.Env
        Entorno del cual se toma el espacio de acciones.

    Retorna
    -------
    action : int
        Acción elegida según la regla simple.
    """
    n_actions = env.action_space.n

    # Acciones esperadas para Space Invaders (según get_action_meanings()):
    # 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
    if n_actions >= 6:
        # Alterna RIGHTFIRE y LEFTFIRE para barrer la pantalla disparando.
        if not hasattr(agente_regla_simple, "_paso"):
            agente_regla_simple._paso = 0
        agente_regla_simple._paso += 1

        ciclo = agente_regla_simple._paso % 10
        if ciclo < 5:
            return 4  # RIGHTFIRE
        else:
            return 5  # LEFTFIRE
    else:
        return env.action_space.sample()


def ejecutar_episodio(env, funcion_agente, max_steps=10000):
    """
    Ejecuta un episodio completo en el entorno usando la función de agente
    dada, hasta que terminated o truncated sea True, o se alcance max_steps.

    Parámetros
    ----------
    env : gymnasium.Env
        Entorno ya creado (posiblemente envuelto en RecordVideo).
    funcion_agente : callable
        Función con firma funcion_agente(observation, env) -> action.
    max_steps : int
        Número máximo de pasos a ejecutar en el episodio.

    Retorna
    -------
    resultado : dict
        Diccionario con al menos:
        - "pasos": int, número de pasos ejecutados.
        - "recompensa_total": float, retorno acumulado del episodio.
        - "terminated": bool
        - "truncated": bool
    """
    observation, info = env.reset()

    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)

        recompensa_total += reward
        pasos += 1

        if terminated or truncated:
            break

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": terminated,
        "truncated": truncated,
    }


def generar_video_agente(nombre_entorno, funcion_agente, video_folder,
                          name_prefix, n_episodios=1, max_steps=10000,
                          **kwargs_entorno):
    """
    Función de alto nivel que combina crear_entorno, ejecutar_episodio y el
    cierre correcto del entorno para generar videos y métricas de un agente.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno de Gymnasium/ALE, p. ej. "ALE/SpaceInvaders-v5".
    funcion_agente : callable
        Función con firma funcion_agente(observation, env) -> action
        (p. ej. agente_aleatorio o agente_regla_simple).
    video_folder : str
        Carpeta donde se guardarán los videos generados.
    name_prefix : str
        Prefijo para los nombres de los archivos de video.
    n_episodios : int
        Número de episodios completos a ejecutar y grabar.
    max_steps : int
        Número máximo de pasos por episodio.
    **kwargs_entorno :
        Argumentos adicionales para gymnasium.make (obs_type, frameskip,
        repeat_action_probability, full_action_space, etc.).

    Retorna
    -------
    resultado : dict
        - "videos": list[str], rutas de los archivos .mp4 generados.
        - "metricas": list[dict], una entrada por episodio con "pasos" y
          "recompensa_total" (y terminated/truncated).
    """
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: True,
        **kwargs_entorno,
    )

    metricas = []
    try:
        for episodio in range(n_episodios):
            resultado_episodio = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps
            )
            metricas.append(resultado_episodio)
    finally:
        # para que RecordVideo guarde los videos
        env.close()

    # Recolectar las rutas de los videos generados en la carpeta.
    videos = []
    if os.path.isdir(video_folder):
        for archivo in sorted(os.listdir(video_folder)):
            if archivo.startswith(name_prefix) and archivo.endswith(".mp4"):
                videos.append(os.path.join(video_folder, archivo))

    return {
        "videos": videos,
        "metricas": metricas,
    }


if __name__ == "__main__":
    # Prueba rápida: un episodio de agente aleatorio en Space Invaders.
    resultado = generar_video_agente(
        nombre_entorno="ALE/SpaceInvaders-v5",
        funcion_agente=agente_aleatorio,
        video_folder="videos_prueba",
        name_prefix="space_invaders_random",
        n_episodios=1,
    )
    print("Videos generados:", resultado["videos"])
    print("Métricas:", resultado["metricas"])
