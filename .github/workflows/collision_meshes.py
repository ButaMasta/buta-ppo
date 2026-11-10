"""Unpacks the COLLISION_MESHES secret (assets/collision_meshes as a base64 .tar.xz, from the MESHES environment
variable) into the directory given as the first argument (default: assets), and sets the step output `available`.

Make the secret from a checkout with the meshes:
    tar -C assets -cf - collision_meshes | xz -9e | base64 -w0 | gh secret set COLLISION_MESHES
"""
import base64
import io
import os
import sys
import tarfile

destination = sys.argv[1] if len(sys.argv) > 1 else "assets"
meshes = os.environ.get("MESHES", "")
available = bool(meshes)
if available:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(meshes)), mode="r:xz") as archive:
        archive.extractall(destination, filter="data")
    print(f"Unpacked the collision meshes into {destination}/collision_meshes.")
else:
    print("::warning::No COLLISION_MESHES secret, so the self-tests are skipped.")

with open(os.environ["GITHUB_OUTPUT"], "a") as output:
    output.write(f"available={'true' if available else 'false'}\n")
