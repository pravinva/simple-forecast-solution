// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useState, useEffect } from "react";
import { Container } from "@material-ui/core";
import styles from "./Header.module.scss";
import classNames from "classnames";
import { Link } from "../../common/link/Link";
import { Logo } from "../../common/panel/logo/Logo";
import { Button } from "../../common/button/Button";

export const Header: React.SFC = () => {
  const [scrollPos, setScrollPos] = useState(0);

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);

    return () => window.removeEventListener("scroll", handleScroll);
  });

  const handleScroll = () => {
    setScrollPos(document.documentElement.scrollTop);
  };

  return (
    <header className={classNames(styles.header, scrollPos > 0 && styles["header--scrolled"])}>
      <Container className="container">
        <Link href="/#">
          <Logo />
        </Link>

        <nav>
          <ul>
            <li>
              <Link href="/#about">About</Link>
            </li>
            <li>
              <Link href="/#how-it-works">How it works</Link>
            </li>
            <li>
              <Link href="/#pricing">Pricing</Link>
            </li>
            <li>
              <Link href="/#faqs">FAQ</Link>
            </li>
            <li>
              <Link href="/authenticated">Login</Link>
            </li>
          </ul>
        </nav>
      </Container>
    </header>
  );
};
