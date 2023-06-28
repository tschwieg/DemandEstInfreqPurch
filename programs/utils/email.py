# -*- coding: utf-8 -*-

from __future__ import division, print_function
from os import linesep
import smtplib

class mailNotify:

    """Send e-mail to self after long program run"""

    def __init__(self,
                 emailFrom = "servers.mcaceresb@gmail.com",
                 emailPass = "DBX/g)]kHz*Vx2j]",
                 serverID  = None):
        """Initialize e-mail, password, and server alias

        Args:
            emailFrom (str): E-mail of sener (i.e. your e-mail)

        Kwargs:
            emailPass (str): Your e-mail password, in plain text (THIS IS
                NOT SAFE! ONLY USE WITH DUMMY ACCOUNTS!)
            serverID (str): Alias to identify the server this is from.
        """

        self.emailFrom = emailFrom
        self.emailPass = emailPass
        self.serverID  = serverID

    def send(self, progStatus, progMessage, dateStart, dateEnd, to = None):
        """Send e-mail to self after long program run
        
        Args:
            progMessage (str): Subject line
            progStatus (str): Program status (OK or stack trace)
            dateStart (str): Start time
            dateEnd (str): End time
            serverID (str): Alias to identify the server this is from
        
        Returns: Sends e-mail to self or recipients in "to"
        """
        
        if to is not None:
            to = ";".join(to) if type(to) is list else to
        else:
            to = self.emailFrom
        
        sub  = "[Automated Message] "
        sub += "" if self.serverID is None else "(server-{0}) ".format(self.serverID)
        sub += progMessage
        msg  = linesep.join([
            "From: " + self.emailFrom,
            "To: " + to,
            "Subject: " + sub,
            "Content-type: text/html",
            "",
            "Program info: <br>",
            "<ul>",
            "<li> Start: " + dateStart + "</li>",
            "<li> End: " + dateEnd + "</li>",
            "<li> Last known status:" + "</li>",
            "</ul>",
            "<pre>",
            progStatus,
            "</pre>"
        ])
        
        username = self.emailFrom
        password = self.emailPass
        server   = smtplib.SMTP('smtp.gmail.com:587')
        
        server.ehlo()
        server.starttls()
        server.login(username,password)
        server.sendmail(username, to, msg)
        server.quit()
