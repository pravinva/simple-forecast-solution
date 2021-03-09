// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import Head from "next/head";
import { Footer } from "../components/common/footer/Footer";
import { Logo } from "../components/common/panel/logo/Logo";
import { Link } from "../components/common/link/Link";
import { Container, Grid } from "@material-ui/core";

const Page = () => {
  return (
      <>
      <Head>
        <title>Amazon Simple Forecast Solution</title>
      </Head>
      <br/><br/>
      <Grid container direction="row" justify="center" alignItems="center" >
          <Logo />
      </Grid>
      <br/><br/>
      <Grid container direction="row" justify="center" alignItems="center" >
      <Link href="/authenticated">
          Login
      </Link>
      </Grid>
      <br/><br/>
      <Footer />
      </>
  );
}

export default Page;
