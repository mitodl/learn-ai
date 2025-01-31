# Keycloak Integration

The Compose file includes a Keycloak instance that you can use for authentication instead of spinning up a separate one or using one of the deployed instances. It's not enabled by default, but you can run it if you prefer not to run your own Keycloak instance.

## Default Settings

There are some defaults that are part of this.

_SSL Certificate_: There's a self-signed cert that's in `config/keycloak/tls` - if you'd rather set up your own (or you have a real cert or something to use), you can drop the PEM files in there. See the README there for info.

_Realm_: There's a `default-realm.json` in `config/keycloak` that will get loaded by Keycloak when it starts up, and will set up a realm for you with some users and a client so you don't have to set it up yourself. The realm it creates is called `ol-local`.

The users it sets up are:

| User                | Password  |
| ------------------- | --------- |
| `student@odl.local` | `student` |
| `prof@odl.local`    | `prof`    |
| `admin@odl.local`   | `admin`   |

The client it sets up is called `apisix`. You can change the passwords and get the secret in the admin.

## Making it Work

If you don't have a Keycloak instance running locally already, you can use the pack-in one. It starts with the rest of the services and is configured to be at `http://kc.ol.local:8006` and `https://kc.ol.local:8007` by default (but you can change this in the `env` files).

Some setup is required to use the pack-in instance:

1. Set required keycloak environment values in your `.env` file:
   - Set a keystore password via `KEYCLOAK_SVC_KEYSTORE_PASSWORD`. This is required, but the password need not be anything special.
   - Set `KEYCLOAK_CLIENT_SECRET`; ask another developer for the relevant value.
2. Optionally add `KEYCLOAK_SVC_HOSTNAME`, `KEYCLOAK_SVC_ADMIN`, and `KEYCLOAK_SVC_ADMIN_PASSWORD` to your `.env` file.
   1. `KEYCLOAK_SVC_HOSTNAME` is the hostname you want to use for the instance - the default is `kc.ol.local`.
   2. `KEYCLOAK_SVC_ADMIN` is the admin username. The default is `admin`.
   3. `KEYCLOAK_SVC_ADMIN_PASSWORD` is the admin password. The default is `admin`.
3. Re-start the stack.

The Keycloak container should start and stay running. Once it does, you should be able to log in at `https://kc.ol.local:8007` with username and password `admin` (or the values you supplied).

If you'd rather use a separate Keycloak instance, ensure these settings are present in the appropriate `env` file (best is probably `backend.local.env`):

- `KEYCLOAK_REALM`

  Sets the realm used by APISIX for Keycloak authentication. Defaults to `ol-local`.

- `KEYCLOAK_DISCOVERY_URL`

  Sets the discovery URL for the Keycloak OIDC service. (In Keycloak admin, navigate to the realm you're using, then go to Realm Settings under Configure, and the link is under OpenID Endpoint Configuration.) This defaults to a valid value for the pack-in Keycloak instance.

- `KEYCLOAK_CLIENT_ID`

  The client ID for the OIDC client for APISIX. Defaults to `apisix`.

- `KEYCLOAK_CLIENT_SECRET`

  The client secret for the OIDC client. No default - you will need to get this from the Keycloak admin, even if you're using the pack-in Keycloak instance.

> If you're using a Keycloak instance also hosted within a Docker container on the same machine you're running the AI chatbots, you'll need to make sure it can be seen from within the `apigateway` container. This will _require_ some work on your part - generally, stuff within Composer environments can't see things outside of their own environment. There's an example of this in the `docker-compose.services.yml` file if your Keycloak instance uses a Compose environment.
