// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useEffect, useState } from "react";
import { Auth } from "@aws-amplify/auth";
import {
  AmplifyAuthenticator,
  AmplifyContainer,
  AmplifySignIn,
  AmplifySignUp,
  AmplifyForgotPassword,
} from "@aws-amplify/ui-react";
import { onAuthUIStateChange, AuthState } from "@aws-amplify/ui-components";
import styles from "./Authenticator.module.scss";
import Head from "next/head";
import { Logo } from "../panel/logo/Logo";
import { Link } from "../link/Link";


export const Authenticator: React.FC = ({ children }) => {
  const [signedIn, setSignedIn] = useState(false);
  const [initialAuthState, setInitialAuthState] = useState<AuthState.SignIn | AuthState.SignUp>(undefined);

  useEffect(() => {
    setInitialAuthState(window.location.search === "?register" ? AuthState.SignUp : AuthState.SignIn);
  }, []);

  useEffect(() => {
    checkIfSignedIn();

    return onAuthUIStateChange((authState) => {
      if (authState === AuthState.SignedIn) {
        setSignedIn(true);
      } else if (authState === AuthState.SignedOut) {
        setSignedIn(false);
      }
    });
  }, []);

  const checkIfSignedIn = async () => {
    try {
      const user = await Auth.currentAuthenticatedUser();
      setSignedIn(!!user);
    } catch (err) {
      console.error(err);
      setSignedIn(false);
    }
  };

  if (!initialAuthState) {
    return <></>;
  }

  if (!signedIn) {
    return (
      <>
        <Head>
          <title>{initialAuthState === AuthState.SignUp ? "Sign Up" : "Login"} - Amazon Simple Forecast Solution</title>
        </Head>
          <AmplifyContainer className={styles.container}>
          <Link className={styles.logo} href="/">
            <Logo />
          </Link>
          <AmplifyAuthenticator initialAuthState={initialAuthState}>
            <AmplifySignIn
              formFields={[
                { type: "email", label: "Email", placeholder: "Email", hint: "Enter your email", required: true },
                {
                  type: "password",
                  label: "Password",
                  placeholder: "Enter your password",
                  required: true,
                },
              ]}
              slot="sign-in"
            />

            <AmplifyForgotPassword
              formFields={[
                { type: "email", label: "Email", placeholder: "Email", hint: "Enter your email", required: true },
              ]}
              slot="forgot-password"
            />
            
            <AmplifySignUp
              usernameAlias="email"
              formFields={[
                { type: "email", label: "Email", placeholder: "Email", hint: "Enter your email", required: true },
                { type: "name", label: "Name", placeholder: "Name", hint: "Enter your name", required: true },
                {
                  type: "custom:company",
                  label: "Company",
                  placeholder: "Company",
                  hint: "Enter your company",
                  required: true,
                },
                {
                  type: "password",
                  label: "Password",
                  placeholder: "Enter your password",
                  required: true,
                },
              ]}
              slot="sign-up"
            />
          </AmplifyAuthenticator>
        </AmplifyContainer>
      </>
    );
  }

  return <>{children}</>;
};

