
from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()

from googleapiclient import discovery
compute = discovery.build('compute', 'v1', credentials=credentials)

def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result['items']
