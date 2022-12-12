import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

'''
* class used to notify the parent.

* the system configurations are initialised  
  at the time of creation of the class object.

* the system then connected to the smtp gmail client  
  over a safe network.  

* the system then sends the locally saved video to the  
  parents.

* the connection with the smtp client is closed.
'''

class Notify:

    def __init__(self):
        # User configuration
        self.sender_email = '' # add sender's email address here
        self.sender_name = '' # add sender's name here
        self.password = '' # add the password key generated
        self.receiver_emails = [] # add the list of parents' email addresses here
        self.receiver_names = [] # add the names of the parents here
        self.filename = 'cam_video.avi' # name of the file stored locally

    def send(self):

        for receiver_email, receiver_name in zip(self.receiver_emails, self.receiver_names):

            print("Connecting to SMTP client...")
            # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
            server = smtplib.SMTP('smtp.gmail.com', 587)
            # Encrypts the email
            print("Encrypting email...")
            context = ssl.create_default_context()
            server.starttls(context=context)
            # We log in into our Google account
            print("Logging in...")
            server.login(self.sender_email, self.password)

            print("Sending the email...")
            # Configurating user's info
            msg = MIMEMultipart()
            msg['To'] = formataddr((receiver_name, receiver_email))
            msg['From'] = formataddr((self.sender_name, self.sender_email))
            msg['Subject'] = 'Hello Parent!'

            msg.attach(
                MIMEText("WARNING: There might have been an occurence of violence..."))

            try:
                # Open PDF file in binary mode
                with open(self.filename, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                # Encode file in ASCII characters to send by email
                encoders.encode_base64(part)

                # Add header as key/value pair to attachment part
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {self.filename}",
                )

                msg.attach(part)
            except Exception as e:
                print(f'Oh no! We didn\'t find the attachment!\n{e}')
                break

            try:

                # Sending email from sender, to receiver with the email body
                server.sendmail(self.sender_email,
                                receiver_email, msg.as_string())
                print('Email sent!')
            except Exception as e:
                print(f'Oh no! Something bad happened!\n{e}')
                break
            finally:
                print('Closing the server...')
                server.quit()
