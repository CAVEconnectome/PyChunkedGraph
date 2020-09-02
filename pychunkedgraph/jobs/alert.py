from typing import Iterable


def send_email(to: Iterable[str], subject: str, message: str) -> None:
    """
    Uses GMail SMTP server to send alerts.
    """
    from os import environ
    from smtplib import SMTP_SSL

    email = environ["ALERT_BOT_EMAIL_ID"]
    password = environ["ALERT_BOT_EMAIL_PASSWORD"]
    text = f"From: AlertBot <{email}>" f"\nSubject: {subject}\n\n{message}"

    server = SMTP_SSL("smtp.gmail.com", 465)
    server.ehlo()
    server.login(email, password)
    server.sendmail(password, to, text)
    server.close()
