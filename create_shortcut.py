"""
Creates a custom app icon (assets/app_icon.ico) and a one-click Desktop & Project Shortcut
('Mood Music Recommender.lnk') on Windows.
"""

import os
import struct
import subprocess
import zlib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
ICON_PATH = os.path.join(ASSETS_DIR, "app_icon.ico")
LAUNCHER_BAT = os.path.join(PROJECT_DIR, "Start_App.bat")


def create_app_icon_ico(ico_path: str):
    """Generate a vibrant 64x64 32-bit PNG-encoded .ico file with a music note & mood glow."""
    os.makedirs(os.path.dirname(ico_path), exist_ok=True)
    w, h = 64, 64
    raw_rows = []

    for y in range(h):
        row = bytearray([0])  # PNG filter type 0
        for x in range(w):
            # Normalized coords [-1, 1]
            nx = (x - 31.5) / 31.5
            ny = (y - 31.5) / 31.5
            dist = (nx * nx + ny * ny) ** 0.5

            if dist > 0.95:
                # Transparent outside rounded badge
                row.extend((0, 0, 0, 0))
                continue

            # Smooth anti-aliased edge
            alpha = 255 if dist < 0.90 else int(255 * (0.95 - dist) / 0.05)

            # Vibrant gradient background (deep purple-blue to cyan-green)
            r = int(40 + 110 * (1 - ny) * 0.5 + 60 * (nx + 1) * 0.5)
            g = int(50 + 160 * (1 - ny) * 0.5)
            b = int(160 + 85 * (nx + 1) * 0.5)

            # Draw white glowing double eighth-note (🎵) in the center
            # Left note head around (22, 43), right note head around (42, 39)
            d_head1 = ((x - 21) ** 2 / 49.0 + (y - 43) ** 2 / 30.0) ** 0.5
            d_head2 = ((x - 41) ** 2 / 49.0 + (y - 39) ** 2 / 30.0) ** 0.5
            stem1 = (25 <= x <= 28) and (18 <= y <= 43)
            stem2 = (45 <= x <= 48) and (14 <= y <= 39)
            # Top beam connecting (25,18) to (48,14)
            beam_y = 18 - (x - 25) * (4.0 / 23.0)
            beam = (25 <= x <= 48) and (beam_y <= y <= beam_y + 5.5)

            if d_head1 <= 1.0 or d_head2 <= 1.0 or stem1 or stem2 or beam:
                r, g, b = 255, 255, 255

            row.extend((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), alpha))
        raw_rows.append(bytes(row))

    raw_data = b"".join(raw_rows)

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw_data, 9))
        + png_chunk(b"IEND", b"")
    )

    # Wrap PNG inside standard ICO container
    ico_header = struct.pack("<HHH", 0, 1, 1)
    ico_entry = struct.pack(
        "<BBBBHHII",
        w,
        h,
        0,
        0,
        1,
        32,
        len(png_bytes),
        6 + 16,
    )
    with open(ico_path, "wb") as f:
        f.write(ico_header + ico_entry + png_bytes)


def create_windows_shortcut(shortcut_path: str, target_bat: str, working_dir: str, icon_path: str):
    """Use Windows Script Host via PowerShell to create a .lnk shortcut with custom icon."""
    ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
$Shortcut.TargetPath = '{target_bat}'
$Shortcut.WorkingDirectory = '{working_dir}'
$Shortcut.IconLocation = '{icon_path},0'
$Shortcut.Description = 'Launch Mood-Based Music Recommender (Auto-Installs Requirements & Opens App)'
$Shortcut.Save()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )


def main():
    create_app_icon_ico(ICON_PATH)
    print(f"[OK] Created application icon: {ICON_PATH}")

    if os.name == "nt":
        # 1. Create shortcut inside project root
        local_lnk = os.path.join(PROJECT_DIR, "Mood Music Recommender.lnk")
        create_windows_shortcut(local_lnk, LAUNCHER_BAT, PROJECT_DIR, ICON_PATH)
        print(f"[OK] Created project app icon: {local_lnk}")

        # 2. Create shortcut on User Desktop
        desktop_dir = os.path.dirname(PROJECT_DIR)
        if os.path.exists(desktop_dir) and "desktop" in desktop_dir.lower():
            desktop_lnk = os.path.join(desktop_dir, "Mood Music Recommender.lnk")
            create_windows_shortcut(desktop_lnk, LAUNCHER_BAT, PROJECT_DIR, ICON_PATH)
            print(f"[OK] Created Desktop app icon: {desktop_lnk}")


if __name__ == "__main__":
    main()
