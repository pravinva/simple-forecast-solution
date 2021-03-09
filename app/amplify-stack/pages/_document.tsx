// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import Document, { Html, Head, Main, NextScript } from "next/document";
import { Typography } from "../components/theme/typography/Typography";

class MyDocument extends Document {
  static async getInitialProps(ctx) {
    const initialProps = await Document.getInitialProps(ctx);
    return { ...initialProps };
  }

  render() {
    return (
      <Html>
          <Head>
          <Typography />
          <link rel="icon" href="/images/logo.svg" type="image/svg+xml"></link>
          </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
