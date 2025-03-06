from play_integrity import Attestation
import structlog
from config import settings
from api.cache.redis_manager import check_and_delete_nonce

logger = structlog.get_logger()


def verify_online(integrity_token):
    attest = Attestation(integrity_token, settings.ANDROID_PACKAGE_NAME)
    data = attest._decrypt_integrity_token()

    if not data:
        return False

    verdicts = []

    token_payload = data.get("tokenPayloadExternal", {})

    # check package name
    request_details = token_payload.get("requestDetails", {})
    verdicts.append(
        request_details.get("requestPackageName") == settings.ANDROID_PACKAGE_NAME
    )

    # check nonce
    verdicts.append(check_and_delete_nonce(request_details.get("requestHash")))

    # device integrity
    deviceIntegrity = token_payload.get("deviceIntegrity", {})
    verdicts.append(
        "MEETS_DEVICE_INTEGRITY" in deviceIntegrity.get("deviceRecognitionVerdict", [])
    )

    # app integrity
    appIntegrity = token_payload["appIntegrity"]
    verdicts.append((bool(appIntegrity.get("certificateSha256Digest"))))
    verdicts.append(
        appIntegrity["appRecognitionVerdict"] == "PLAY_RECOGNIZED"
        or settings.FLASK_ENV == "development"
    )

    # accountDetails
    accountDetails = token_payload["accountDetails"]
    verdicts.append(
        accountDetails.get("appLicensingVerdict") == "LICENSED"
        or settings.FLASK_ENV == "development"
    )

    print("integrity verdicts: " + str(verdicts))
    logger.debug("integrity verdicts: " + str(verdicts))
    return all(verdicts)
