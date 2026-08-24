"""
VENLA V0.1
Google Colab Training Launcher

Fungsi:
- Clone/update repository
- Install dependency
- Check GPU/CUDA
- Read Supabase credentials dari Colab Secrets
- Menjalankan model test
- Menyiapkan environment untuk training

PENTING:
Secret Supabase TIDAK disimpan di GitHub.
"""

import os
import sys
import subprocess


# ============================================================
# CONFIGURATION
# ============================================================

REPOSITORY_URL = (
    "https://github.com/"
    "Mangono1/"
    "venla_robot.git"
)

PROJECT_DIR = "/content/venla_robot"


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_command(
    command,
    cwd=None,
):
    """
    Menjalankan command shell dan menghentikan proses
    jika command gagal.
    """

    print()
    print("=" * 60)
    print("COMMAND:")
    print(command)
    print("=" * 60)
    print()

    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command gagal:\n"
            + command
        )

    return result


# ============================================================
# REPOSITORY
# ============================================================

def prepare_repository():
    """
    Clone repository jika belum ada.

    Jika repository sudah ada, ambil update terbaru.
    """

    print("=" * 60)
    print("VENLA COLAB - REPOSITORY")
    print("=" * 60)
    print()

    if not os.path.exists(
        PROJECT_DIR
    ):

        print(
            "Repository belum ada."
        )

        print(
            "Clone GitHub..."
        )

        run_command(
            "git clone "
            + REPOSITORY_URL
            + " "
            + PROJECT_DIR
        )

    else:

        print(
            "Repository sudah ada:"
        )

        print(
            PROJECT_DIR
        )

        print()

        print(
            "Mengambil update terbaru..."
        )

        run_command(
            "git pull",
            cwd=PROJECT_DIR,
        )

    print()

    print(
        "Repository siap."
    )


# ============================================================
# DEPENDENCY
# ============================================================

def install_dependencies():
    """
    Install dependency dari requirements.txt.
    """

    print("=" * 60)
    print("VENLA COLAB - DEPENDENCIES")
    print("=" * 60)
    print()

    requirements_path = os.path.join(
        PROJECT_DIR,
        "requirements.txt",
    )

    if not os.path.exists(
        requirements_path
    ):

        print(
            "WARNING:"
        )

        print(
            "requirements.txt belum tersedia."
        )

        print(
            "Installation dilewati."
        )

        return

    print(
        "Installing requirements..."
    )

    run_command(

        sys.executable
        + " -m pip install -q -r requirements.txt",

        cwd=PROJECT_DIR,

    )

    print()

    print(
        "Dependency siap."
    )


# ============================================================
# HARDWARE CHECK
# ============================================================

def hardware_test():
    """
    Mengecek PyTorch, CUDA, GPU dan VRAM.
    """

    print("=" * 60)
    print("VENLA COLAB - HARDWARE CHECK")
    print("=" * 60)
    print()

    import torch

    print(
        "PyTorch :",
        torch.__version__
    )

    print(
        "CUDA    :",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )

        print(
            "GPU     :",
            gpu_name
        )

        properties = (
            torch.cuda.get_device_properties(
                0
            )
        )

        vram_gb = (
            properties.total_memory
            / (
                1024 ** 3
            )
        )

        print(
            "VRAM    :",
            f"{vram_gb:.2f} GB"
        )

        print()

        print(
            "CUDA GPU AKTIF."
        )

    else:

        print()

        print(
            "WARNING:"
        )

        print(
            "CUDA tidak aktif."
        )

        print(
            "Training akan menggunakan CPU."
        )

    print()


# ============================================================
# SUPABASE
# ============================================================

def setup_supabase():
    """
    Membaca credential Supabase.

    Prioritas:

    1. Environment variable
    2. Google Colab Secrets

    Tidak pernah menulis secret ke file.
    """

    print("=" * 60)
    print("VENLA COLAB - SUPABASE")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # ENVIRONMENT VARIABLE
    # --------------------------------------------------------

    existing_url = os.environ.get(
        "SUPABASE_URL"
    )

    existing_key = os.environ.get(
        "SUPABASE_KEY"
    )

    if (
        existing_url
        and
        existing_key
    ):

        print(
            "Supabase credential ditemukan "
            "dari environment."
        )

        print(
            "URL:",
            existing_url
        )

        print(
            "KEY: ********"
        )

        return True


    # --------------------------------------------------------
    # GOOGLE COLAB SECRETS
    # --------------------------------------------------------

    try:

        from google.colab import userdata

        print(
            "Mencoba membaca Google Colab Secrets..."
        )

        try:

            supabase_url = (
                userdata.get(
                    "SUPABASE_URL"
                )
            )

        except Exception:

            supabase_url = None


        try:

            supabase_key = (
                userdata.get(
                    "SUPABASE_KEY"
                )
            )

        except Exception:

            supabase_key = None


        if (
            supabase_url
            and
            supabase_key
        ):

            os.environ[
                "SUPABASE_URL"
            ] = supabase_url

            os.environ[
                "SUPABASE_KEY"
            ] = supabase_key

            print()

            print(
                "Supabase Secret berhasil dibaca."
            )

            print(
                "URL:",
                supabase_url
            )

            print(
                "KEY: ********"
            )

            return True


    except Exception as error:

        print(
            "Google Colab Secrets tidak tersedia."
        )

        print(
            "Detail:",
            error
        )


    # --------------------------------------------------------
    # NO CREDENTIAL
    # --------------------------------------------------------

    print()

    print(
        "WARNING:"
    )

    print(
        "Supabase credential belum tersedia."
    )

    print()

    print(
        "Training masih dapat dijalankan,"
    )

    print(
        "tetapi sinkronisasi Supabase "
        "belum aktif."
    )

    print()

    return False


# ============================================================
# MODEL TEST
# ============================================================

def model_test():
    """
    Menjalankan test VENLA model.
    """

    print("=" * 60)
    print("VENLA COLAB - MODEL TEST")
    print("=" * 60)
    print()

    model_path = os.path.join(
        PROJECT_DIR,
        "venla",
        "model.py",
    )

    if not os.path.exists(
        model_path
    ):

        print(
            "WARNING:"
        )

        print(
            "venla/model.py belum tersedia."
        )

        print(
            "Model test dilewati."
        )

        return False


    run_command(

        sys.executable
        + " -m venla.model",

        cwd=PROJECT_DIR,

    )

    return True


# ============================================================
# TRAINING
# ============================================================

def start_training(
    steps=1000,
):
    """
    Menjalankan training VENLA.

    Fungsi ini sengaja tidak dipanggil otomatis
    pada tahap initialization.
    """

    print("=" * 60)
    print("VENLA COLAB - TRAINING")
    print("=" * 60)
    print()

    train_path = os.path.join(
        PROJECT_DIR,
        "venla",
        "train.py",
    )

    if not os.path.exists(
        train_path
    ):

        raise FileNotFoundError(

            "venla/train.py belum dibuat. "

            "Selesaikan trainer terlebih dahulu."

        )


    command = (

        sys.executable

        + " -m venla.train"

        + " --steps "

        + str(
            int(steps)
        )

    )


    run_command(

        command,

        cwd=PROJECT_DIR,

    )


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

def show_environment():
    """
    Menampilkan informasi environment VENLA.
    """

    print("=" * 60)
    print("VENLA ENVIRONMENT")
    print("=" * 60)
    print()

    print(
        "Python:",
        sys.version.split()[0]
    )

    print(
        "Project:",
        PROJECT_DIR
    )

    print(
        "Repository:",
        REPOSITORY_URL
    )

    print()

    print(
        "Environment siap."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 60)
    print("VENLA V0.1")
    print("GOOGLE COLAB TRAINING ENVIRONMENT")
    print("=" * 60)

    print()


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    prepare_repository()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    install_dependencies()


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    if PROJECT_DIR not in sys.path:

        sys.path.insert(
            0,
            PROJECT_DIR,
        )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    show_environment()


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    hardware_test()


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    supabase_ready = (
        setup_supabase()
    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    model_ready = (
        model_test()
    )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()

    print("=" * 60)
    print("VENLA COLAB INITIALIZATION")
    print("=" * 60)

    print()

    print(
        "Repository :",
        "OK"
    )

    print(
        "Hardware   :",
        "OK"
    )

    print(
        "Supabase   :",
        "OK"
        if supabase_ready
        else "BELUM TERHUBUNG"
    )

    print(
        "Model      :",
        "OK"
        if model_ready
        else "BELUM TERSEDIA"
    )

    print()

    print(
        "Training BELUM dijalankan."
    )

    print()

    print(
        "Setelah seluruh engine selesai,"
    )

    print(
        "training dapat dipanggil dengan:"
    )

    print()

    print(
        "start_training(1000)"
    )

    print()

    print("=" * 60)
    print("VENLA COLAB INITIALIZATION SELESAI")
    print("=" * 60)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
