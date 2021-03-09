// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import { useEffect } from "react";
import { configureAmplify } from "../services/amplify";
import { VoyagerTheme } from "../components/theme/VoyagerTheme";
import { Authenticator } from "../components/common/auth/Authenticator";
import "../components/theme/typography/Typography.scss";
import "../components/theme/reset/Reset.scss";
import "../components/theme/scroll/Scroll.scss";

const App = ({ Component, pageProps }) => {
  useEffect(configureAmplify, []);

  // Force authentication wrapper on all pages under /authenticated
  if (typeof window !== "undefined" && window.location.pathname.startsWith("/authenticated")) {
    return (
      <Authenticator>
        <Component {...pageProps} />
      </Authenticator>
    );
  }

  return <Component {...pageProps} />;
};

const ThemedApp = (props) => (
  <VoyagerTheme>
    <App {...props} />
  </VoyagerTheme>
);

export default ThemedApp;
