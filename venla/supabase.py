"""
VENLA V0.1
Supabase Storage Client

Fungsi:
- koneksi Supabase
- upload artifact
- download artifact
- delete artifact
- cek keberadaan file
- list file
- metadata artifact

Credential TIDAK disimpan di source code.

Environment variables:
    SUPABASE_URL
    SUPABASE_KEY

SUPABASE_KEY dapat berupa:
    Publishable key
    Legacy anon/public key

Jangan gunakan service_role/secret key
di source code atau repository publik.
"""

import os
import json
import urllib.request
import urllib.error
import urllib.parse


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BUCKET = "venla"


# ============================================================
# EXCEPTIONS
# ============================================================

class SupabaseError(Exception):
    """Base exception untuk Supabase."""

    pass


class SupabaseConfigurationError(
    SupabaseError
):
    """Credential Supabase belum lengkap."""

    pass


class SupabaseRequestError(
    SupabaseError
):
    """Request Supabase gagal."""

    pass


# ============================================================
# CLIENT
# ============================================================

class VENLASupabase:
    """
    Client HTTP sederhana untuk Supabase.

    Tidak membutuhkan package tambahan.

    Menggunakan urllib bawaan Python.
    """

    def __init__(
        self,
        url=None,
        key=None,
        bucket=DEFAULT_BUCKET,
    ):

        self.url = (
            url
            or os.environ.get(
                "SUPABASE_URL"
            )
        )

        self.key = (
            key
            or os.environ.get(
                "SUPABASE_KEY"
            )
        )

        self.bucket = bucket


        # ----------------------------------------------------
        # NORMALIZE URL
        # ----------------------------------------------------

        if self.url:

            self.url = self.url.rstrip(
                "/"
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not self.url:

            raise SupabaseConfigurationError(
                "SUPABASE_URL belum tersedia."
            )

        if not self.key:

            raise SupabaseConfigurationError(
                "SUPABASE_KEY belum tersedia."
            )


    # ========================================================
    # HEADERS
    # ========================================================

    def _headers(
        self,
        content_type=None,
    ):

        headers = {
            "apikey": self.key,
            "Authorization":
                "Bearer " + self.key,
        }

        if content_type:

            headers[
                "Content-Type"
            ] = content_type

        return headers


    # ========================================================
    # REQUEST
    # ========================================================

    def _request(
        self,
        method,
        url,
        data=None,
        content_type=None,
        timeout=60,
    ):

        request = urllib.request.Request(
            url=url,
            data=data,
            headers=self._headers(
                content_type
            ),
            method=method,
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:

                body = response.read()

                return (
                    response.status,
                    body,
                    dict(
                        response.headers
                    ),
                )


        except urllib.error.HTTPError as error:

            body = error.read()

            try:

                body_text = body.decode(
                    "utf-8",
                    errors="replace",
                )

            except Exception:

                body_text = str(
                    body
                )

            raise SupabaseRequestError(
                f"HTTP {error.code}: "
                f"{body_text}"
            ) from error


        except urllib.error.URLError as error:

            raise SupabaseRequestError(
                "Koneksi Supabase gagal: "
                + str(error)
            ) from error


    # ========================================================
    # CONNECTION TEST
    # ========================================================

    def test_connection(self):

        """
        Test koneksi dengan Storage API.

        Tidak membutuhkan tabel database.
        """

        url = (
            self.url
            + "/storage/v1/bucket"
        )

        status, body, headers = (
            self._request(
                "GET",
                url,
            )
        )

        return {
            "status":
                status,

            "connected":
                200 <= status < 300,

            "body":
                body.decode(
                    "utf-8",
                    errors="replace",
                ),
        }


    # ========================================================
    # CREATE BUCKET
    # ========================================================

    def create_bucket(
        self,
        bucket=None,
        public=False,
    ):

        """
        Membuat bucket.

        CATATAN:
        Publishable/anon key biasanya tidak memiliki
        permission untuk membuat bucket.

        Jika gagal 401/403, bucket harus dibuat
        dari dashboard Supabase.
        """

        bucket = (
            bucket
            or self.bucket
        )

        url = (
            self.url
            + "/storage/v1/bucket"
        )

        payload = json.dumps(
            {
                "id": bucket,
                "name": bucket,
                "public": bool(
                    public
                ),
            }
        ).encode(
            "utf-8"
        )

        status, body, headers = (
            self._request(
                "POST",
                url,
                data=payload,
                content_type=(
                    "application/json"
                ),
            )
        )

        return {
            "status":
                status,

            "bucket":
                bucket,

            "body":
                body.decode(
                    "utf-8",
                    errors="replace",
                ),
        }


    # ========================================================
    # UPLOAD
    # ========================================================

    def upload_file(
        self,
        local_path,
        remote_path,
        bucket=None,
        content_type=(
            "application/octet-stream"
        ),
        overwrite=True,
    ):

        """
        Upload file ke Supabase Storage.

        Parameters:
            local_path:
                path file lokal.

            remote_path:
                path di bucket.

            bucket:
                nama bucket.

            overwrite:
                jika True gunakan endpoint update.
        """

        bucket = (
            bucket
            or self.bucket
        )

        if not os.path.exists(
            local_path
        ):

            raise FileNotFoundError(
                "File lokal tidak ditemukan: "
                + local_path
            )


        with open(
            local_path,
            "rb",
        ) as file:

            data = file.read()


        remote_path = remote_path.lstrip(
            "/"
        )


        encoded_path = "/".join(
            urllib.parse.quote(
                part,
                safe="",
            )
            for part
            in remote_path.split("/")
        )


        if overwrite:

            method = "POST"

            url = (
                self.url
                + "/storage/v1/object/"
                + urllib.parse.quote(
                    bucket,
                    safe="",
                )
                + "/"
                + encoded_path
            )

            # Supabase Storage menerima upsert
            # melalui header ini.

            headers = self._headers(
                content_type
            )

            headers[
                "x-upsert"
            ] = "true"

            request = urllib.request.Request(
                url=url,
                data=data,
                headers=headers,
                method=method,
            )

        else:

            url = (
                self.url
                + "/storage/v1/object/"
                + urllib.parse.quote(
                    bucket,
                    safe="",
                )
                + "/"
                + encoded_path
            )

            request = urllib.request.Request(
                url=url,
                data=data,
                headers=self._headers(
                    content_type
                ),
                method="POST",
            )


        try:

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                body = response.read()

                return {
                    "status":
                        response.status,

                    "bucket":
                        bucket,

                    "path":
                        remote_path,

                    "size":
                        len(data),

                    "response":
                        body.decode(
                            "utf-8",
                            errors="replace",
                        ),
                }


        except urllib.error.HTTPError as error:

            body = error.read()

            text = body.decode(
                "utf-8",
                errors="replace",
            )

            raise SupabaseRequestError(
                "Upload gagal. "
                f"HTTP {error.code}: {text}"
            ) from error


    # ========================================================
    # DOWNLOAD
    # ========================================================

    def download_file(
        self,
        remote_path,
        local_path,
        bucket=None,
    ):

        """
        Download artifact dari Supabase Storage.
        """

        bucket = (
            bucket
            or self.bucket
        )

        remote_path = remote_path.lstrip(
            "/"
        )

        encoded_path = "/".join(
            urllib.parse.quote(
                part,
                safe="",
            )
            for part
            in remote_path.split("/")
        )

        url = (
            self.url
            + "/storage/v1/object/"
            + urllib.parse.quote(
                bucket,
                safe="",
            )
            + "/"
            + encoded_path
        )


        status, body, headers = (
            self._request(
                "GET",
                url,
                timeout=300,
            )
        )


        directory = os.path.dirname(
            os.path.abspath(
                local_path
            )
        )

        os.makedirs(
            directory,
            exist_ok=True
        )


        with open(
            local_path,
            "wb",
        ) as file:

            file.write(
                body
            )


        return {
            "status":
                status,

            "bucket":
                bucket,

            "path":
                remote_path,

            "local_path":
                local_path,

            "size":
                len(body),
        }


    # ========================================================
    # DELETE
    # ========================================================

    def delete_file(
        self,
        remote_path,
        bucket=None,
    ):

        """
        Delete file dari Storage.
        """

        bucket = (
            bucket
            or self.bucket
        )

        remote_path = remote_path.lstrip(
            "/"
        )

        encoded_path = "/".join(
            urllib.parse.quote(
                part,
                safe="",
            )
            for part
            in remote_path.split("/")
        )

        url = (
            self.url
            + "/storage/v1/object/"
            + urllib.parse.quote(
                bucket,
                safe="",
            )
        )

        payload = json.dumps(
            [
                remote_path
            ]
        ).encode(
            "utf-8"
        )


        status, body, headers = (
            self._request(
                "DELETE",
                url,
                data=payload,
                content_type=(
                    "application/json"
                ),
            )
        )


        return {
            "status":
                status,

            "bucket":
                bucket,

            "path":
                remote_path,

            "response":
                body.decode(
                    "utf-8",
                    errors="replace",
                ),
        }


    # ========================================================
    # LIST FILES
    # ========================================================

    def list_files(
        self,
        prefix="",
        bucket=None,
        limit=100,
        offset=0,
    ):

        """
        List object di bucket.
        """

        bucket = (
            bucket
            or self.bucket
        )

        url = (
            self.url
            + "/storage/v1/object/list/"
            + urllib.parse.quote(
                bucket,
                safe="",
            )
        )


        payload = json.dumps(
            {
                "prefix":
                    prefix,

                "limit":
                    int(limit),

                "offset":
                    int(offset),

                "sortBy":
                    {
                        "column":
                            "name",

                        "order":
                            "asc",
                    },
            }
        ).encode(
            "utf-8"
        )


        status, body, headers = (
            self._request(
                "POST",
                url,
                data=payload,
                content_type=(
                    "application/json"
                ),
            )
        )


        try:

            objects = json.loads(
                body.decode(
                    "utf-8"
                )
            )

        except Exception:

            objects = []


        return objects


    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        remote_path,
        bucket=None,
    ):

        """
        Cek apakah object tersedia.
        """

        bucket = (
            bucket
            or self.bucket
        )

        remote_path = remote_path.lstrip(
            "/"
        )

        encoded_path = "/".join(
            urllib.parse.quote(
                part,
                safe="",
            )
            for part
            in remote_path.split("/")
        )

        url = (
            self.url
            + "/storage/v1/object/info/"
            + urllib.parse.quote(
                bucket,
                safe="",
            )
            + "/"
            + encoded_path
        )


        try:

            status, body, headers = (
                self._request(
                    "GET",
                    url,
                )
            )

            return True

        except SupabaseRequestError:

            return False


    # ========================================================
    # BUCKET
    # ========================================================

    def set_bucket(
        self,
        bucket,
    ):

        self.bucket = bucket

        return self.bucket


# ============================================================
# FACTORY
# ============================================================

def create_supabase_client(
    bucket=DEFAULT_BUCKET,
):

    return VENLASupabase(
        bucket=bucket
    )


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():

    url = os.environ.get(
        "SUPABASE_URL"
    )

    key = os.environ.get(
        "SUPABASE_KEY"
    )

    return {
        "url_available":
            bool(url),

        "key_available":
            bool(key),

        "ready":
            bool(url and key),
    }


# ============================================================
# TEST
# ============================================================

def test_supabase():

    print("=" * 60)
    print("VENLA - SUPABASE CONNECTION TEST")
    print("=" * 60)

    print()

    environment = check_environment()

    print(
        "SUPABASE_URL:",
        environment[
            "url_available"
        ]
    )

    print(
        "SUPABASE_KEY:",
        environment[
            "key_available"
        ]
    )

    print()

    if not environment[
        "ready"
    ]:

        print(
            "⚠️ Credential belum tersedia."
        )

        print()

        print(
            "Set environment:"
        )

        print(
            "SUPABASE_URL"
        )

        print(
            "SUPABASE_KEY"
        )

        print()

        return False


    client = create_supabase_client()

    print(
        "Supabase URL:",
        client.url
    )

    print(
        "Bucket:",
        client.bucket
    )

    print()

    try:

        result = client.test_connection()

        print(
            "HTTP Status:",
            result[
                "status"
            ]
        )

        print(
            "✅ SUPABASE TERHUBUNG"
        )

        print()

        return True

    except Exception as error:

        print(
            "❌ SUPABASE CONNECTION ERROR"
        )

        print(
            str(error)
        )

        print()

        return False


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_supabase()
